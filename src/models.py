"""
Fast model ladder:
L0 - Deterministic baseline
L1 - Logistic Regression
L2 - LightGBM

All models use:
fit(X, y)
predict_proba(X)
save(path)
load(path)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb

from src.utils import get_config, Timer
from src.features import FEATURE_NAMES


def _cfg() -> dict:
    return get_config()["model"]


# ============================================================
# L0 - Deterministic Baseline
# ============================================================

class DeterministicBaseline:

    _FI = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeterministicBaseline":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        fi = self._FI

        name_score = (
            0.4 * X[:, fi["name_exact"]]
            + 0.2 * X[:, fi["name_edit_sim"]]
            + 0.2 * X[:, fi["name_token_set_sim"]]
            + 0.2 * X[:, fi["name_token_jaccard"]]
        )

        addr_score = (
            0.3 * X[:, fi["addr_exact"]]
            + 0.2 * X[:, fi["addr_edit_sim"]]
            + 0.2 * X[:, fi["addr_token_set_sim"]]
            + 0.2 * X[:, fi["digit_overlap"]]
            + 0.1 * X[:, fi["digit_exact"]]
        )

        penalty = (
            0.3 * X[:, fi["contr_same_name_diff_addr"]]
            + 0.3 * X[:, fi["contr_same_addr_diff_name"]]
            + 0.2 * X[:, fi["contr_digit_conflict"]]
            + 0.2 * X[:, fi["contr_country_conflict"]]
        )

        score = (
            0.5 * name_score
            + 0.5 * addr_score
            - 0.3 * penalty
        )

        score = np.clip(score, 0.0, 1.0)

        return np.column_stack([
            1.0 - score,
            score
        ])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "DeterministicBaseline":
        return pickle.loads(path.read_bytes())


# ============================================================
# L1 - Logistic Regression
# ============================================================

class LRModel:

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 300,
        seed: int = 20260925
    ) -> None:

        self.C = C
        self.max_iter = max_iter
        self.seed = seed

        self._model: Pipeline | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> "LRModel":

        cfg = _cfg()

        print("\n[L1] Training Logistic Regression...")
        print(f"[L1] Training rows: {len(X):,}")
        print(f"[L1] Features: {X.shape[1]}")

        with Timer("LR fit"):

            self._model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=cfg.get("lr_C", self.C),
                        max_iter=min(
                            cfg.get("lr_max_iter", 300),
                            300
                        ),
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=cfg.get(
                            "random_seed",
                            self.seed
                        ),
                        n_jobs=-1
                    )
                )
            ])

            self._model.fit(X, y)

        print("[L1] Logistic Regression training complete.")

        return self

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        if self._model is None:
            raise RuntimeError(
                "LRModel has not been trained."
            )

        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_bytes(
            pickle.dumps(self)
        )

    @classmethod
    def load(
        cls,
        path: Path
    ) -> "LRModel":

        return pickle.loads(
            path.read_bytes()
        )


# ============================================================
# L2 - FAST LIGHTGBM
# ============================================================

class LGBMModel:

    def __init__(self) -> None:

        self._model: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> "LGBMModel":

        cfg = _cfg()

        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))

        scale = neg / max(pos, 1)

        print("\n[L2] Training FAST LightGBM...")
        print(f"[L2] Training rows : {len(X):,}")
        print(f"[L2] Features      : {X.shape[1]:,}")
        print(f"[L2] Positive      : {pos:,}")
        print(f"[L2] Negative      : {neg:,}")
        print(f"[L2] Scale weight  : {scale:.2f}")

        # ----------------------------------------------------
        # FAST SETTINGS
        # ----------------------------------------------------

        n_estimators = min(
            cfg.get("lgbm_n_estimators", 150),
            150
        )

        num_leaves = min(
            cfg.get("lgbm_num_leaves", 31),
            31
        )

        learning_rate = max(
            cfg.get("lgbm_learning_rate", 0.10),
            0.08
        )

        print("\n[L2] Parameters:")
        print(f"     n_estimators = {n_estimators}")
        print(f"     learning_rate = {learning_rate}")
        print(f"     num_leaves = {num_leaves}")
        print("     n_jobs = 4")

        with Timer("LightGBM FAST fit"):

            self._model = lgb.LGBMClassifier(

                # FAST
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,

                # Reduce tree complexity
                max_depth=-1,
                min_child_samples=30,

                # Sampling
                subsample=0.8,
                colsample_bytree=0.8,

                # Regularization
                reg_alpha=0.1,
                reg_lambda=0.1,

                # Imbalance
                scale_pos_weight=scale,

                # Reproducibility
                random_state=cfg.get(
                    "random_seed",
                    20260925
                ),

                # IMPORTANT:
                # Do not use every CPU core.
                # Your machine has only 16 GB RAM.
                n_jobs=4,

                verbosity=-1
            )

            self._model.fit(
                X,
                y,
                feature_name=FEATURE_NAMES
            )

        print("\n[L2] LightGBM training complete.")

        return self

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        if self._model is None:
            raise RuntimeError(
                "LGBMModel has not been trained."
            )

        return self._model.predict_proba(X)

    def get_feature_importance(
        self
    ) -> pd.DataFrame:

        if self._model is None:
            raise RuntimeError(
                "LGBMModel has not been trained."
            )

        return (
            pd.DataFrame({
                "feature": FEATURE_NAMES,
                "importance":
                    self._model.feature_importances_
            })
            .sort_values(
                "importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

    def save(
        self,
        path: Path
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_bytes(
            pickle.dumps(self)
        )

    @classmethod
    def load(
        cls,
        path: Path
    ) -> "LGBMModel":

        return pickle.loads(
            path.read_bytes()
        )