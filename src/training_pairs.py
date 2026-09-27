"""
Leakage-safe training pair construction.

Split strategy:
    S1-entity-level stratified split.
    All pairs for one S1 stay in the same partition.

Negative categories:
    1. Random negatives
    2. Same-name / different-address hard negatives
    3. Same-address / different-name hard negatives
    4. Exact-name / exact-address candidate hard negatives

Designed for large datasets and limited RAM.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


# ============================================================
# S1 TRAIN / VALIDATION SPLIT
# ============================================================

def stratified_s1_split(
    gt: dict[str, list[str]],
    val_fraction: float = 0.15,
    seed: int = 20260925,
) -> tuple[list[str], list[str]]:
    """
    Split S1 IDs into train / validation.

    Stratification groups:
        zero matches
        single match
        multiple matches

    All pairs belonging to one S1 remain in the same partition.
    """

    rng = random.Random(seed)

    zero = []
    single = []
    multi = []

    for s1_id, matches in gt.items():

        n = len(matches)

        if n == 0:
            zero.append(s1_id)

        elif n == 1:
            single.append(s1_id)

        else:
            multi.append(s1_id)

    train_ids = []
    val_ids = []

    for group in (zero, single, multi):

        rng.shuffle(group)

        if len(group) <= 1:
            n_val = 0
        else:
            n_val = max(
                1,
                int(len(group) * val_fraction)
            )

            # Never put the complete group into validation.
            n_val = min(
                n_val,
                len(group) - 1
            )

        val_ids.extend(group[:n_val])
        train_ids.extend(group[n_val:])

    print(
        f"[split] Train S1: {len(train_ids):,} | "
        f"Validation S1: {len(val_ids):,}"
    )

    return train_ids, val_ids


# ============================================================
# TRAINING PAIRS
# ============================================================

def build_training_pairs(
    candidates_df: pd.DataFrame,
    gt: dict[str, list[str]],
    train_s1_ids: set[str],
    s1_lookup: dict[str, tuple],
    cand_lookup: dict[str, tuple],
    neg_ratio_random: int = 3,
    neg_ratio_hard: int = 3,
    max_hard_per_s1: int = 10,
    seed: int = 20260925,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build training pairs.

    Positive:
        candidate_id exists in ground truth for the S1.

    Negative:
        candidate_id does not exist in ground truth.

    Only training S1 entities are used.

    Returns:
        pairs_df
        labels
    """

    rng = random.Random(seed)

    train_s1_set = set(train_s1_ids)

    # --------------------------------------------------------
    # Ground truth sets
    # --------------------------------------------------------

    gt_sets = {
        s1_id: set(matches)
        for s1_id, matches in gt.items()
    }

    # --------------------------------------------------------
    # Keep only training S1 candidates
    # --------------------------------------------------------

    mask = candidates_df["s1_id"].isin(train_s1_set)

    train_cands = candidates_df.loc[
        mask,
        [
            "s1_id",
            "candidate_id",
            "channels",
            "n_channels",
        ],
    ]

    print(
        f"[pairs] Candidate rows for training: "
        f"{len(train_cands):,}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    positive_rows = []
    negative_rows = []

    # --------------------------------------------------------
    # Group by S1
    # --------------------------------------------------------

    for s1_id, group in train_cands.groupby(
        "s1_id",
        sort=False,
    ):

        true_set = gt_sets.get(
            s1_id,
            set()
        )

        if not true_set:
            # ------------------------------------------------
            # Zero-match S1
            # ------------------------------------------------

            negative_indices = list(
                range(len(group))
            )

            if not negative_indices:
                continue

            n_random = min(
                neg_ratio_random,
                len(negative_indices)
            )

            selected = rng.sample(
                negative_indices,
                n_random
            )

            for idx in selected:

                row = group.iloc[idx]

                negative_rows.append({
                    "s1_id": s1_id,
                    "candidate_id": row["candidate_id"],
                    "channels": row["channels"],
                    "n_channels": row["n_channels"],
                    "label": 0,
                })

            continue

        # ----------------------------------------------------
        # Candidate IDs
        # ----------------------------------------------------

        candidate_ids = group["candidate_id"].tolist()

        positive_ids = [
            cid
            for cid in candidate_ids
            if cid in true_set
        ]

        negative_ids = [
            cid
            for cid in candidate_ids
            if cid not in true_set
        ]

        # ----------------------------------------------------
        # Build candidate lookup ONLY for this S1
        # ----------------------------------------------------

        group_by_candidate = {
            row["candidate_id"]: row
            for row in group.to_dict("records")
        }

        # ----------------------------------------------------
        # Positives
        # ----------------------------------------------------

        for cid in positive_ids:

            row = group_by_candidate[cid]

            positive_rows.append({
                "s1_id": s1_id,
                "candidate_id": cid,
                "channels": row["channels"],
                "n_channels": row["n_channels"],
                "label": 1,
            })

        if not negative_ids:
            continue

        # ----------------------------------------------------
        # S1 information
        # ----------------------------------------------------

        s1_info = s1_lookup.get(
            s1_id,
            ("", "", "", "")
        )

        s1_name = s1_info[0]
        s1_addr = s1_info[1]

        hard_pool = []
        random_pool = []

        # ----------------------------------------------------
        # Classify negatives
        # ----------------------------------------------------

        for cid in negative_ids:

            row = group_by_candidate[cid]

            cand_info = cand_lookup.get(
                cid,
                ("", "", "", "")
            )

            cand_name = cand_info[0]
            cand_addr = cand_info[1]

            channels = row["channels"]

            # Safely handle None / NaN
            if channels is None:
                channels_text = ""

            elif isinstance(channels, (list, tuple, set)):
                channels_text = " ".join(
                    str(x)
                    for x in channels
                )

            else:
                channels_text = str(
                    channels
                )

            same_name = (
                bool(s1_name)
                and bool(cand_name)
                and s1_name == cand_name
                and s1_addr != cand_addr
            )

            same_address = (
                bool(s1_addr)
                and bool(cand_addr)
                and s1_addr == cand_addr
                and s1_name != cand_name
            )

            exact_name = (
                "exact_name"
                in channels_text
            )

            exact_address = (
                "exact_address"
                in channels_text
            )

            is_hard = (
                same_name
                or same_address
                or exact_name
                or exact_address
            )

            record = {
                "s1_id": s1_id,
                "candidate_id": cid,
                "channels": row["channels"],
                "n_channels": row["n_channels"],
                "label": 0,
            }

            if is_hard:
                hard_pool.append(record)
            else:
                random_pool.append(record)

        # ----------------------------------------------------
        # Number of positives
        # ----------------------------------------------------

        n_pos = len(positive_ids)

        # ----------------------------------------------------
        # Hard negatives
        # ----------------------------------------------------

        n_hard = min(
            n_pos * neg_ratio_hard,
            max_hard_per_s1,
            len(hard_pool),
        )

        if n_hard > 0:

            selected = rng.sample(
                hard_pool,
                n_hard
            )

            negative_rows.extend(
                selected
            )

        # ----------------------------------------------------
        # Random negatives
        # ----------------------------------------------------

        n_random = min(
            n_pos * neg_ratio_random,
            len(random_pool),
        )

        if n_random > 0:

            selected = rng.sample(
                random_pool,
                n_random
            )

            negative_rows.extend(
                selected
            )

    # ========================================================
    # Combine
    # ========================================================

    all_rows = (
        positive_rows
        + negative_rows
    )

    if not all_rows:

        raise RuntimeError(
            "No training pairs were generated. "
            "Check candidate generation and ground truth."
        )

    pairs_df = pd.DataFrame(
        all_rows
    )

    labels = (
        pairs_df["label"]
        .to_numpy(
            dtype=np.int8
        )
    )

    print(
        f"\n[pairs] Training pairs: "
        f"{len(pairs_df):,}"
    )

    print(
        f"[pairs] Positives: "
        f"{int(labels.sum()):,}"
    )

    print(
        f"[pairs] Negatives: "
        f"{int((labels == 0).sum()):,}"
    )

    return pairs_df, labels


# ============================================================
# VALIDATION PAIRS
# ============================================================

def build_validation_pairs(
    candidates_df: pd.DataFrame,
    gt: dict[str, list[str]],
    val_s1_ids: set[str],
) -> pd.DataFrame:
    """
    Build validation pairs.

    All candidate rows belonging to validation S1 IDs
    are retained and labelled using ground truth.
    """

    val_s1_set = set(
        val_s1_ids
    )

    gt_sets = {
        s1_id: set(matches)
        for s1_id, matches in gt.items()
    }

    # --------------------------------------------------------
    # Filter validation candidates
    # --------------------------------------------------------

    mask = candidates_df["s1_id"].isin(
        val_s1_set
    )

    val_cands = candidates_df.loc[
        mask
    ].copy()

    if val_cands.empty:

        raise RuntimeError(
            "No validation candidates found."
        )

    # --------------------------------------------------------
    # Create labels without apply()
    # --------------------------------------------------------

    val_cands["label"] = [
        int(
            candidate_id
            in gt_sets.get(
                s1_id,
                set()
            )
        )
        for s1_id, candidate_id
        in zip(
            val_cands["s1_id"],
            val_cands["candidate_id"]
        )
    ]

    positive_count = int(
        val_cands["label"].sum()
    )

    print(
        f"[validation] Pairs: "
        f"{len(val_cands):,}"
    )

    print(
        f"[validation] Positives: "
        f"{positive_count:,}"
    )

    print(
        f"[validation] Negatives: "
        f"{len(val_cands) - positive_count:,}"
    )

    return val_cands