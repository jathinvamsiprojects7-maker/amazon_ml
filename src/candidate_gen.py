"""
Memory-safe candidate generation for the Amazon ML entity matching project.

Designed for machines with limited RAM.

Channels used in fast/safe mode:
    1. exact_name
    2. exact_address
    3. address_digits

Important:
    - Does NOT build a huge in-memory inverted index.
    - Does NOT build a multi-million-row TF-IDF matrix.
    - Does NOT call .toarray() on a large sparse similarity matrix.
    - Uses SQLite indexes on disk.
    - Retrieves candidates in small S1 chunks.
    - Caps candidates per channel and per S1.

The pipeline imports:
    SourceIndex
    retrieve_candidates_streaming
    merge_candidate_parquets
    measure_candidate_recall
"""

from __future__ import annotations

import gc
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils import get_config, Timer


# ============================================================================
# CHANNEL DEFINITIONS
# ============================================================================

CAND_SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()),
        ("candidate_id", pa.string()),
        ("source", pa.string()),
        ("channels", pa.string()),
        ("n_channels", pa.int16()),
    ]
)

CHANNEL_BITS = {
    "exact_name": 1,
    "exact_address": 2,
    "address_digits": 4,
    "rare_token": 8,
    "ngram": 16,
}


def bits_to_channels(bits: int) -> str:
    parts = []

    for name, bit in CHANNEL_BITS.items():
        if bits & bit:
            parts.append(name)

    return "|".join(sorted(parts))


def number_of_channels(bits: int) -> int:
    return int(bits).bit_count()


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def _retrieval_cfg() -> dict[str, Any]:
    """
    Read retrieval configuration while supporting both the old and new
    configuration names used in your project.
    """

    cfg = get_config()
    rcfg = cfg.get("retrieval", {})

    def first(*names, default=None):
        for name in names:
            if name in rcfg:
                return rcfg[name]
        return default

    return {
        # Final maximum candidates retained for one S1.
        "per_s1": int(
            first(
                "per_s1",
                default=1000,
            )
        ),

        # Exact channels.
        "exact_cap": int(
            first(
                "exact_cap",
                "max_bucket_size",
                default=200,
            )
        ),

        # Address digit channel.
        "digit_cap": int(
            first(
                "digit_cap",
                default=200,
            )
        ),

        # Retrieval chunk.
        "retrieval_chunk_size": int(
            first(
                "retrieval_chunk_size",
                default=5000,
            )
        ),

        # Output buffer.
        "output_buffer_pairs": int(
            first(
                "output_buffer_pairs",
                default=100000,
            )
        ),

        # These are retained for compatibility.
        "min_token_len": int(
            first(
                "min_token_len",
                "rare_token_min_len",
                default=3,
            )
        ),

        "max_token_df": first(
            "max_token_df",
            "rare_token_max_df",
            default=1000,
        ),

        "ngram_size": int(
            first(
                "ngram_size",
                "ngram_n",
                default=3,
            )
        ),

        "ngram_top_k": int(
            first(
                "ngram_top_k",
                default=50,
            )
        ),

        "ngram_max_features": int(
            first(
                "ngram_max_features",
                default=30000,
            )
        ),

        # IMPORTANT:
        # Disabled by default because your machine has only ~16 GB RAM.
        "use_ngram": bool(
            first(
                "use_ngram",
                default=False,
            )
        ),
    }


# ============================================================================
# DISK-BACKED SOURCE INDEX
# ============================================================================

class SourceIndex:
    """
    Disk-backed index for one source.

    Instead of storing millions of Python dictionaries in RAM, the source
    records are stored in SQLite with normal B-tree indexes.

    This is slower than a huge in-memory index, but it is dramatically safer
    on a 16 GB machine.
    """

    def __init__(self, source_label: str):
        self.source_label = source_label

        self.db_path: Path | None = None
        self.conn: sqlite3.Connection | None = None

        self._built = False
        self._row_count = 0

    # ----------------------------------------------------------------------
    # SQLite setup
    # ----------------------------------------------------------------------

    def _open_database(self, db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(db_path),
            timeout=120,
        )

        # Speed-oriented settings.
        # Data can be rebuilt from parquet, so durability during index
        # construction is less important than avoiding excessive I/O.
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-100000")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")

        return conn

    def _database_is_valid(
        self,
        conn: sqlite3.Connection,
        parquet_path: Path,
    ) -> bool:

        try:
            tables = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table'
                    """
                ).fetchall()
            }

            if "records" not in tables:
                return False

            row = conn.execute(
                "SELECT COUNT(*) FROM records"
            ).fetchone()

            if row is None:
                return False

            count = int(row[0])

            if count <= 0:
                return False

            return True

        except Exception:
            return False

    # ----------------------------------------------------------------------
    # Build SQLite index
    # ----------------------------------------------------------------------

    def build_from_parquet(
        self,
        parquet_path: Path,
        rare_token_min_len: int = 3,
        rare_token_max_df: float | int = 0.01,
        ngram_n: int = 3,
        ngram_max_features: int = 30000,

        # New configuration names.
        min_token_len: int | None = None,
        max_token_df: float | int | None = None,
        ngram_size: int | None = None,
    ) -> None:

        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Normalized parquet file not found:\n{parquet_path}"
            )

        cfg = _retrieval_cfg()

        if min_token_len is not None:
            rare_token_min_len = min_token_len

        if max_token_df is not None:
            rare_token_max_df = max_token_df

        if ngram_size is not None:
            ngram_n = ngram_size

        # One SQLite file per source.
        db_path = parquet_path.with_suffix(
            parquet_path.suffix
            + f".{self.source_label.lower()}.idx.sqlite"
        )

        self.db_path = db_path

        print(
            f"  [{self.source_label}] "
            f"Disk-backed index: {db_path.name}"
        )

        # --------------------------------------------------------------
        # Reuse an existing index.
        # --------------------------------------------------------------

        if db_path.exists():
            try:
                conn = self._open_database(db_path)

                if self._database_is_valid(conn, parquet_path):
                    self.conn = conn
                    self._row_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM records"
                        ).fetchone()[0]
                    )
                    self._built = True

                    print(
                        f"  [{self.source_label}] "
                        f"Existing SQLite index reused: "
                        f"{self._row_count:,} rows"
                    )

                    return

                conn.close()

            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass

            try:
                db_path.unlink()
            except Exception:
                pass

        # --------------------------------------------------------------
        # Create new database.
        # --------------------------------------------------------------

        print(
            f"  [{self.source_label}] "
            f"Creating disk-backed SQLite index..."
        )

        conn = self._open_database(db_path)

        self.conn = conn

        conn.execute(
            """
            CREATE TABLE records (
                row_id INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL,
                name_norm TEXT,
                address_norm TEXT,
                address_dig TEXT
            )
            """
        )

        conn.commit()

        parquet_file = pq.ParquetFile(str(parquet_path))

        insert_sql = """
            INSERT INTO records (
                row_id,
                entity_id,
                name_norm,
                address_norm,
                address_dig
            )
            VALUES (?, ?, ?, ?, ?)
        """

        row_offset = 0
        t0 = time.perf_counter()

        # Keep each pandas batch small.
        arrow_batch_size = 100_000

        for batch_no, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=arrow_batch_size,
                columns=[
                    "entity_id",
                    "name_norm",
                    "address_norm",
                    "address_dig",
                ],
            ),
            start=1,
        ):

            df = batch.to_pandas()

            for col in (
                "entity_id",
                "name_norm",
                "address_norm",
                "address_dig",
            ):
                if col not in df.columns:
                    df[col] = ""

                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                )

            rows = []

            for i, row in enumerate(
                df.itertuples(index=False),
            ):
                rows.append(
                    (
                        row_offset + i,
                        row.entity_id,
                        row.name_norm,
                        row.address_norm,
                        row.address_dig,
                    )
                )

            conn.executemany(insert_sql, rows)

            row_offset += len(rows)

            if batch_no % 5 == 0:
                conn.commit()

                elapsed = time.perf_counter() - t0

                print(
                    f"  [{self.source_label}] "
                    f"indexed {row_offset:,} rows | "
                    f"{elapsed:.1f}s"
                )

            del rows
            del df
            gc.collect()

        conn.commit()

        self._row_count = row_offset

        print(
            f"  [{self.source_label}] "
            f"Creating SQLite indexes..."
        )

        # --------------------------------------------------------------
        # Create indexes AFTER inserting all rows.
        # This is much faster than maintaining them during insertion.
        # --------------------------------------------------------------

        with Timer(
            f"SQLite indexes [{self.source_label}]"
        ):

            conn.execute(
                """
                CREATE INDEX idx_records_name
                ON records(name_norm)
                """
            )

            conn.execute(
                """
                CREATE INDEX idx_records_address
                ON records(address_norm)
                """
            )

            conn.execute(
                """
                CREATE INDEX idx_records_digits
                ON records(address_dig)
                """
            )

            conn.commit()

        # Analyze query planner statistics.
        conn.execute("ANALYZE")
        conn.commit()

        self._built = True

        elapsed = time.perf_counter() - t0

        print(
            f"  [{self.source_label}] "
            f"SQLite index ready: "
            f"{self._row_count:,} rows in {elapsed:.1f}s"
        )

        # These are intentionally NOT built.
        print(
            f"  [{self.source_label}] "
            f"Rare-token index: DISABLED in fast-safe mode"
        )

        print(
            f"  [{self.source_label}] "
            f"N-gram index: DISABLED in fast-safe mode"
        )

    # ----------------------------------------------------------------------
    # Batch query helper
    # ----------------------------------------------------------------------

    def _require_connection(self) -> sqlite3.Connection:
        if not self._built or self.conn is None:
            raise RuntimeError(
                f"{self.source_label} SourceIndex has not been built."
            )

        return self.conn

    def _create_query_table(
        self,
        s1_df: pd.DataFrame,
    ) -> None:

        conn = self._require_connection()

        conn.execute("DROP TABLE IF EXISTS temp.query_s1")

        conn.execute(
            """
            CREATE TEMP TABLE query_s1 (
                row_id INTEGER PRIMARY KEY,
                s1_id TEXT NOT NULL,
                name_norm TEXT,
                address_norm TEXT,
                address_dig TEXT
            )
            """
        )

        rows = []

        for i, row in enumerate(
            s1_df[
                [
                    "entity_id",
                    "name_norm",
                    "address_norm",
                    "address_dig",
                ]
            ].itertuples(index=False)
        ):

            entity_id = (
                ""
                if pd.isna(row.entity_id)
                else str(row.entity_id)
            )

            name_norm = (
                ""
                if pd.isna(row.name_norm)
                else str(row.name_norm)
            )

            address_norm = (
                ""
                if pd.isna(row.address_norm)
                else str(row.address_norm)
            )

            address_dig = (
                ""
                if pd.isna(row.address_dig)
                else str(row.address_dig)
            )

            rows.append(
                (
                    i,
                    entity_id,
                    name_norm,
                    address_norm,
                    address_dig,
                )
            )

        conn.executemany(
            """
            INSERT INTO temp.query_s1 (
                row_id,
                s1_id,
                name_norm,
                address_norm,
                address_dig
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

        conn.commit()

        del rows

    # ----------------------------------------------------------------------
    # Exact channel retrieval
    # ----------------------------------------------------------------------

    def _query_channel(
        self,
        column_name: str,
        cap: int,
    ) -> list[tuple[str, str]]:

        conn = self._require_connection()

        sql = f"""
            SELECT s1_id, entity_id
            FROM (
                SELECT
                    q.row_id AS q_row_id,
                    q.s1_id AS s1_id,
                    r.entity_id AS entity_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY q.row_id
                        ORDER BY r.row_id
                    ) AS rn
                FROM temp.query_s1 q
                INNER JOIN records r
                    ON r.{column_name} = q.{column_name}
                WHERE
                    q.{column_name} IS NOT NULL
                    AND q.{column_name} <> ''
                    AND r.{column_name} IS NOT NULL
                    AND r.{column_name} <> ''
            )
            WHERE rn <= ?
        """

        return conn.execute(
            sql,
            (int(cap),),
        ).fetchall()

    def query_batch(
        self,
        s1_df: pd.DataFrame,
        exact_cap: int = 200,
        digit_cap: int = 200,
        per_s1: int = 1000,
    ) -> dict[tuple[str, str], int]:

        conn = self._require_connection()

        self._create_query_table(s1_df)

        pair_bits: dict[tuple[str, str], int] = {}

        # --------------------------------------------------------------
        # Exact name
        # --------------------------------------------------------------

        name_rows = self._query_channel(
            "name_norm",
            exact_cap,
        )

        for s1_id, candidate_id in name_rows:

            key = (
                str(s1_id),
                str(candidate_id),
            )

            pair_bits[key] = (
                pair_bits.get(key, 0)
                | CHANNEL_BITS["exact_name"]
            )

        # --------------------------------------------------------------
        # Exact address
        # --------------------------------------------------------------

        address_rows = self._query_channel(
            "address_norm",
            exact_cap,
        )

        for s1_id, candidate_id in address_rows:

            key = (
                str(s1_id),
                str(candidate_id),
            )

            pair_bits[key] = (
                pair_bits.get(key, 0)
                | CHANNEL_BITS["exact_address"]
            )

        # --------------------------------------------------------------
        # Address digits
        # --------------------------------------------------------------

        digit_rows = self._query_channel(
            "address_dig",
            digit_cap,
        )

        for s1_id, candidate_id in digit_rows:

            key = (
                str(s1_id),
                str(candidate_id),
            )

            pair_bits[key] = (
                pair_bits.get(key, 0)
                | CHANNEL_BITS["address_digits"]
            )

        # --------------------------------------------------------------
        # Enforce final per-S1 candidate budget.
        # --------------------------------------------------------------

        if per_s1 > 0:

            grouped: dict[
                str,
                list[tuple[str, int]]
            ] = defaultdict(list)

            for (s1_id, candidate_id), bits in pair_bits.items():

                grouped[s1_id].append(
                    (
                        candidate_id,
                        bits,
                    )
                )

            limited: dict[tuple[str, str], int] = {}

            for s1_id, candidates in grouped.items():

                if len(candidates) <= per_s1:

                    for candidate_id, bits in candidates:
                        limited[
                            (s1_id, candidate_id)
                        ] = bits

                    continue

                # Rank candidates using channel evidence.
                #
                # More independent channels = stronger candidate.
                # Exact name/address receive extra weight.
                def candidate_score(item):
                    candidate_id, bits = item

                    score = (
                        number_of_channels(bits) * 100
                    )

                    if bits & CHANNEL_BITS["exact_name"]:
                        score += 30

                    if bits & CHANNEL_BITS["exact_address"]:
                        score += 30

                    if bits & CHANNEL_BITS["address_digits"]:
                        score += 10

                    return (
                        -score,
                        candidate_id,
                    )

                candidates.sort(
                    key=candidate_score
                )

                for candidate_id, bits in candidates[
                    :per_s1
                ]:
                    limited[
                        (s1_id, candidate_id)
                    ] = bits

            pair_bits = limited

        conn.execute("DROP TABLE IF EXISTS temp.query_s1")
        conn.commit()

        return pair_bits

    # ----------------------------------------------------------------------
    # Close
    # ----------------------------------------------------------------------

    def close(self) -> None:

        if self.conn is not None:

            try:
                self.conn.close()
            except Exception:
                pass

            self.conn = None

        self._built = False

        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================================================================
# STREAMING RETRIEVAL
# ============================================================================

def retrieve_candidates_streaming(
    s1_df: pd.DataFrame,
    source_idx: SourceIndex,
    output_path: Path,
    chunk_size: int = 5000,
    ngram_batch_size: int = 2000,
    ngram_top_k: int = 50,
    max_bucket: int = 200,
) -> int:
    """
    Generate candidates from one source.

    This function intentionally avoids:
        - a giant pair_bitmap
        - full candidate DataFrame
        - ngram dense matrices

    Candidate rows are buffered and written to parquet periodically.
    """

    cfg = _retrieval_cfg()

    # Use config values when available.
    chunk_size = int(
        cfg.get(
            "retrieval_chunk_size",
            chunk_size,
        )
    )

    exact_cap = int(
        cfg.get(
            "exact_cap",
            max_bucket,
        )
    )

    digit_cap = int(
        cfg.get(
            "digit_cap",
            max_bucket,
        )
    )

    per_s1 = int(
        cfg.get(
            "per_s1",
            1000,
        )
    )

    output_buffer_pairs = int(
        cfg.get(
            "output_buffer_pairs",
            100000,
        )
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Never append to a partially written candidate file.
    if output_path.exists():
        output_path.unlink()

    n_total = len(s1_df)

    total_pairs = 0

    t_start = time.perf_counter()

    writer: pq.ParquetWriter | None = None

    # Output buffers.
    out_s1 = []
    out_candidate = []
    out_source = []
    out_channels = []
    out_n_channels = []

    def flush_output() -> None:
        nonlocal writer

        if not out_s1:
            return

        table = pa.table(
            {
                "s1_id": pa.array(
                    out_s1,
                    type=pa.string(),
                ),
                "candidate_id": pa.array(
                    out_candidate,
                    type=pa.string(),
                ),
                "source": pa.array(
                    out_source,
                    type=pa.string(),
                ),
                "channels": pa.array(
                    out_channels,
                    type=pa.string(),
                ),
                "n_channels": pa.array(
                    out_n_channels,
                    type=pa.int16(),
                ),
            },
            schema=CAND_SCHEMA,
        )

        if writer is None:

            writer = pq.ParquetWriter(
                str(output_path),
                schema=CAND_SCHEMA,
                compression="snappy",
            )

        writer.write_table(table)

        del table

        out_s1.clear()
        out_candidate.clear()
        out_source.clear()
        out_channels.clear()
        out_n_channels.clear()

        gc.collect()

    # --------------------------------------------------------------
    # Process S1 in chunks.
    # --------------------------------------------------------------

    for chunk_start in range(
        0,
        n_total,
        chunk_size,
    ):

        chunk_end = min(
            chunk_start + chunk_size,
            n_total,
        )

        chunk = s1_df.iloc[
            chunk_start:chunk_end
        ].copy()

        pair_bits = source_idx.query_batch(
            chunk,
            exact_cap=exact_cap,
            digit_cap=digit_cap,
            per_s1=per_s1,
        )

        for (s1_id, candidate_id), bits in pair_bits.items():

            out_s1.append(str(s1_id))
            out_candidate.append(str(candidate_id))
            out_source.append(
                source_idx.source_label
            )
            out_channels.append(
                bits_to_channels(bits)
            )
            out_n_channels.append(
                number_of_channels(bits)
            )

        total_pairs += len(pair_bits)

        # Flush frequently.
        if len(out_s1) >= output_buffer_pairs:
            flush_output()

        elapsed = (
            time.perf_counter()
            - t_start
        )

        processed = chunk_end

        rate = (
            processed
            / max(elapsed, 0.001)
        )

        remaining = n_total - processed

        eta_seconds = (
            remaining
            / max(rate, 0.001)
        )

        print(
            f"  [{source_idx.source_label}] "
            f"{processed:,}/{n_total:,} S1s | "
            f"pairs={total_pairs:,} | "
            f"{rate:.0f} S1/s | "
            f"ETA={eta_seconds / 60:.1f}m"
        )

        del pair_bits
        del chunk

        gc.collect()

    flush_output()

    if writer is not None:
        writer.close()

    elapsed = (
        time.perf_counter()
        - t_start
    )

    print(
        f"  [{source_idx.source_label}] "
        f"Total candidate pairs: "
        f"{total_pairs:,}"
    )

    print(
        f"  [{source_idx.source_label}] "
        f"Retrieval time: "
        f"{elapsed / 60:.2f} minutes"
    )

    return total_pairs


# ============================================================================
# MERGE S2 + S3 CANDIDATES
# ============================================================================

def merge_candidate_parquets(
    path_a: Path,
    path_b: Path,
    output_path: Path,
) -> int:
    """
    Memory-safe merge of S2 and S3 candidate parquet files.

    Because S2 and S3 candidate IDs are source-specific, we normally don't
    need a huge in-memory concat/drop_duplicates operation.
    """

    path_a = Path(path_a)
    path_b = Path(path_b)
    output_path = Path(output_path)

    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None

    total = 0

    def append_file(path: Path) -> None:
        nonlocal writer
        nonlocal total

        if not path.exists():
            return

        parquet_file = pq.ParquetFile(
            str(path)
        )

        if writer is None:

            writer = pq.ParquetWriter(
                str(output_path),
                schema=CAND_SCHEMA,
                compression="snappy",
            )

        for batch in parquet_file.iter_batches(
            batch_size=100_000
        ):

            table = pa.Table.from_batches(
                [batch],
                schema=CAND_SCHEMA,
            )

            writer.write_table(table)

            total += batch.num_rows

            del table

        gc.collect()

    append_file(path_a)
    append_file(path_b)

    if writer is not None:
        writer.close()

    print(
        f"  Merged candidate pairs: "
        f"{total:,}"
    )

    return total


# ============================================================================
# CANDIDATE RECALL
# ============================================================================

def measure_candidate_recall(
    candidates_df: pd.DataFrame,
    gt: dict[str, list[str]],
    s1_ids: list[str],
) -> dict[str, Any]:
    """
    Measure how many ground-truth matches exist inside the candidate set.
    """

    s1_set = set(s1_ids)

    gt_sets = {
        s1: set(matches)
        for s1, matches in gt.items()
        if s1 in s1_set
    }

    candidate_sets: dict[
        str,
        set[str]
    ] = defaultdict(set)

    subset = candidates_df[
        candidates_df["s1_id"].isin(s1_set)
    ]

    for row in subset.itertuples(
        index=False
    ):

        candidate_sets[
            row.s1_id
        ].add(
            row.candidate_id
        )

    total_gt = 0
    total_recalled = 0

    counts_per_s1 = []

    for s1_id, true_set in gt_sets.items():

        if not true_set:
            continue

        recalled = len(
            true_set
            & candidate_sets.get(
                s1_id,
                set(),
            )
        )

        total_gt += len(true_set)
        total_recalled += recalled

        counts_per_s1.append(
            len(
                candidate_sets.get(
                    s1_id,
                    set(),
                )
            )
        )

    recall = (
        total_recalled / total_gt
        if total_gt > 0
        else 0.0
    )

    counts_arr = (
        np.array(
            counts_per_s1,
            dtype=np.int64,
        )
        if counts_per_s1
        else np.array(
            [0],
            dtype=np.int64,
        )
    )

    result = {
        "candidate_recall": round(
            recall,
            6,
        ),
        "total_gt_positives": int(
            total_gt
        ),
        "total_recalled": int(
            total_recalled
        ),
        "n_s1_with_matches": len(
            counts_per_s1
        ),
        "mean_candidates_per_s1": round(
            float(
                counts_arr.mean()
            ),
            1,
        ),
        "p50_candidates": int(
            np.percentile(
                counts_arr,
                50,
            )
        ),
        "p95_candidates": int(
            np.percentile(
                counts_arr,
                95,
            )
        ),
        "p99_candidates": int(
            np.percentile(
                counts_arr,
                99,
            )
        ),
        "total_pairs": int(
            len(subset)
        ),
    }

    print()
    print(
        "Candidate recall:"
    )
    print(
        f"  Recall: "
        f"{result['candidate_recall']:.4f}"
    )
    print(
        f"  GT positives: "
        f"{result['total_gt_positives']:,}"
    )
    print(
        f"  Recalled: "
        f"{result['total_recalled']:,}"
    )
    print(
        f"  Mean candidates/S1: "
        f"{result['mean_candidates_per_s1']:.1f}"
    )
    print(
        f"  P95 candidates/S1: "
        f"{result['p95_candidates']:,}"
    )

    return result