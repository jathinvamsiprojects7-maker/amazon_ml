# Amazon ML Challenge 2026
# Amazon ML Challenge 2026

Business entity resolution for matching records in Source 1 to equivalent
entities in Source 2 and Source 3.

## Pipeline

```text
Raw TSV files
   -> normalization and parquet caches
   -> exact/token/digit candidate retrieval
   -> candidate union and deduplication
   -> pair feature matrix
   -> training pairs and S1-level validation split
   -> L0/L1/L2 model ladder
   -> hard-negative mining and LightGBM retraining
   -> test inference
   -> matching_results.tsv and candidate_pairs.tsv
   -> official submission validator
```

The main orchestration entry point is `run_pipeline.py`.

## Required Inputs

Place the official challenge files under:

```text
dataset/student_resource/dataset/
   train/
      train_source1.tsv
      train_source2.tsv
      train_source3.tsv
      train_ground_truth.tsv
   test/
      test_source1.tsv
      test_source2.tsv
      test_source3.tsv
```

Each source TSV must contain:

```text
entity_id
business_name
business_address
country
```

The training ground-truth TSV must contain `source1_entity_id` and
`matched_entity_ids`. The second field is a comma-separated list; an empty
value means that the Source 1 record has no known match.

Configured paths are in [config/pipeline.yaml](config/pipeline.yaml). They
currently use absolute `D:/amazon_ml/...` paths and must be updated if the
repository is moved.

## Normalization

Normalization performs script-safe Unicode normalization, case folding,
punctuation/whitespace normalization, and address digit extraction.

```powershell
python run_normalize.py --train
python run_normalize.py --test
python run_normalize.py --all
```

Generated caches:

```text
cache/train_source1_norm.parquet
cache/train_source2_norm.parquet
cache/train_source3_norm.parquet
cache/test_source1_norm.parquet
cache/test_source2_norm.parquet
cache/test_source3_norm.parquet
```

Normalized parquet adds `name_norm`, `address_norm`, `address_dig`,
`name_tokens_str`, and `address_tokens_str` to the raw source columns.

## Candidate Retrieval

Retrieval uses exact normalized name, exact normalized address, address digits,
and rare-token postings. Run one source at a time:

```powershell
python run_retrieval.py --split train --source S2
python run_retrieval.py --split train --source S3
python run_retrieval.py --split test --source S2
python run_retrieval.py --split test --source S3
```

For limited RAM, use one worker and smaller batches:

```powershell
python run_retrieval.py --split test --source S2 `
   --workers 1 --chunk 2000 --index-batch 25000 --max-df 100
```

Candidate outputs are `cache/{train,test}_cand_S2.parquet` and
`cache/{train,test}_cand_S3.parquet`. Rows contain `s1_row`, `cand_row`, and
the provenance field `channels`.

Always check candidate recall before trusting model metrics. A model cannot
recover a true match that retrieval did not include.

## Features

Run after candidate generation:

```powershell
python run_features.py train
python run_features.py test
```

Outputs:

```text
cache/train_features.npy
cache/train_features.meta.json
cache/test_features.npy
cache/test_features.meta.json
```

The feature matrix is float32 with 41 features covering name, address, digit,
country, cross-field, contradiction, missingness, retrieval provenance, and
channel-count signals.

## Training

```powershell
python run_pipeline.py
```

The runner loads data, creates an S1-grouped train/validation split, builds
positive and negative pairs, computes features, trains models, tunes an
S1-level Macro F0.5 threshold, mines hard negatives, and retrains LightGBM.

Available model artifacts:

```text
models/l0_deterministic.pkl   Deterministic similarity baseline
models/l1_logistic.pkl        Logistic regression
models/l2_lgbm.pkl            LightGBM
models/l2_lgbm_hn.pkl         LightGBM with hard negatives
models/final_model.pkl        Selected final model
models/final_config.json      Selected model and threshold
```

Current selected model: `L2+HN`, validation Macro F0.5 `0.062228`, threshold
`0.6658`.

## Reports

Important reports include:

```text
reports/candidate_recall_val.json
reports/model_comparison.json
reports/error_analysis_round1.json
reports/error_analysis_round2.json
reports/final_val_summary.json
reports/lgbm_feature_importance.csv
reports/pipeline_summary.json
```

`model_comparison.json` contains threshold, Macro F0.5, precision, recall,
and zero-match accuracy for L0, L1, L2, and L2+HN.

## Test Outputs

After test candidates and test features exist, inference writes:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

`matching_results.tsv` contains:

```text
source1_entity_id    matched_entity_ids
```

`candidate_pairs.tsv` contains:

```text
source1_entity_id    candidate_entity_ids
```

The official validator is:

```text
dataset/student_resource/utils/validate_submission.py
```

A completely successful run also creates `reports/pipeline_summary.json` with
`validator_pass: true`.

## Configuration and Resources

`config/pipeline.yaml` controls paths, normalization, retrieval caps,
training ratios, model parameters, threshold search, RAM ceiling, CPU share,
and worker count. The threshold grid is currently 20 points because 100-point
S1-level validation was too slow on this machine.

The project processes millions of rows and requires approximately 16 GB RAM,
substantial free disk, and a recent Python environment. Dependencies include
pandas, NumPy, PyArrow, scikit-learn, LightGBM, RapidFuzz, PyYAML, and psutil.
Avoid running multiple retrieval or pipeline processes simultaneously.

## Limitations

- Current measured training candidate recall is approximately 0.45%, making
   retrieval the main quality bottleneck.
- Full test retrieval may exceed available RAM with default settings.
- Lowering `--max-df`, index batch size, or chunk size reduces memory use but
   can also reduce recall or increase runtime.
- The trained model artifacts are complete, but submission outputs are not
   complete until both output TSVs pass the official validator.

## Verification Checklist

```powershell
Get-Process python
Get-ChildItem models
Get-Content reports/model_comparison.json
Get-Content models/final_config.json
Get-ChildItem output
Get-Content reports/pipeline_summary.json
```

Training is complete when model files and `model_comparison.json` exist. The
full project run is complete only when `matching_results.tsv`,
`candidate_pairs.tsv`, and a passing `pipeline_summary.json` exist.
