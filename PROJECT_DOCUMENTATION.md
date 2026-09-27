# Amazon ML Challenge 2026 Project Documentation

## Purpose

This project performs business entity resolution: it matches records from Source 1 to equivalent records in Source 2 and Source 3.

The pipeline is:

```text
Raw TSV input
 -> normalization and parquet caches
 -> candidate retrieval
 -> candidate features
 -> model training and validation
 -> hard-negative mining
 -> test inference
 -> submission TSV files
 -> official validation
```

## Input Files

Place the official data here:

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

Each source TSV requires:

```text
entity_id
business_name
business_address
country
```

The training ground-truth file requires:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` is a comma-separated list. Empty values mean no known match.

Paths are configured in `config/pipeline.yaml`.

## Main Commands

Normalize data:

```powershell
python run_normalize.py --train
python run_normalize.py --test
python run_normalize.py --all
```

Generate candidates:

```powershell
python run_retrieval.py --split train --source S2
python run_retrieval.py --split train --source S3
python run_retrieval.py --split test --source S2
python run_retrieval.py --split test --source S3
```

For limited RAM:

```powershell
python run_retrieval.py --split test --source S2 `
  --workers 1 --chunk 2000 --index-batch 25000 --max-df 100
```

Generate features:

```powershell
python run_features.py train
python run_features.py test
```

Train and validate models:

```powershell
python run_pipeline.py
```

## Generated Caches

Normalization caches:

```text
cache/train_source1_norm.parquet
cache/train_source2_norm.parquet
cache/train_source3_norm.parquet
cache/test_source1_norm.parquet
cache/test_source2_norm.parquet
cache/test_source3_norm.parquet
```

Candidate caches:

```text
cache/train_cand_S2.parquet
cache/train_cand_S3.parquet
cache/test_cand_S2.parquet
cache/test_cand_S3.parquet
```

Feature caches:

```text
cache/train_features.npy
cache/train_features.meta.json
cache/test_features.npy
cache/test_features.meta.json
```

Normalized rows contain the original fields plus `name_norm`, `address_norm`,
`address_dig`, `name_tokens_str`, and `address_tokens_str`.

## Retrieval Signals

Candidate retrieval uses:

- exact normalized name
- exact normalized address
- address digits
- rare informative tokens
- retrieval provenance and channel budgets

Candidate rows contain `s1_row`, `cand_row`, and `channels`.
Candidate recall must be checked before trusting model metrics because a model
cannot recover a match that retrieval did not generate.

## Features

The feature matrix has 41 float32 features covering:

- name exact and fuzzy similarity
- address exact and fuzzy similarity
- digit overlap and sequence similarity
- country match
- cross-field similarity
- contradiction indicators
- missingness
- retrieval provenance
- number of retrieval channels

## Models

The project trains these model variants:

```text
models/l0_deterministic.pkl
models/l1_logistic.pkl
models/l2_lgbm.pkl
models/l2_lgbm_hn.pkl
models/final_model.pkl
models/final_config.json
```

Model meanings:

| Model | Meaning |
|---|---|
| L0 | Deterministic weighted similarity baseline |
| L1 | Logistic regression |
| L2 | LightGBM classifier |
| L2+HN | LightGBM retrained with hard negatives |

The selected model is currently `L2+HN`.

Current validation values:

```text
Selected model: L2+HN
Threshold: 0.6658
Validation Macro F0.5: 0.062228
```

## Reports

Important reports:

```text
reports/candidate_recall_val.json
reports/model_comparison.json
reports/error_analysis_round1.json
reports/error_analysis_round2.json
reports/final_val_summary.json
reports/lgbm_feature_importance.csv
reports/pipeline_summary.json
```

`model_comparison.json` contains thresholds and validation metrics for L0, L1,
L2, and L2+HN.

## Final Output Files

A complete test inference run writes:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

`matching_results.tsv` schema:

```text
source1_entity_id    matched_entity_ids
```

`candidate_pairs.tsv` schema:

```text
source1_entity_id    candidate_entity_ids
```

The official validator is:

```text
dataset/student_resource/utils/validate_submission.py
```

A fully completed run should also create:

```text
reports/pipeline_summary.json
```

with `validator_pass: true`.

## Configuration

All major settings are in `config/pipeline.yaml`:

- dataset and output paths
- normalization behavior
- retrieval caps and token limits
- train/validation split
- negative sampling
- model parameters
- threshold search
- RAM and CPU limits

The threshold search currently uses 20 points because 100-point S1-level
validation was too slow on the available machine.

## Resource Requirements

The project processes millions of records. Recommended minimums:

- approximately 16 GB RAM
- substantial free disk space
- recent Python 3.x
- pandas, NumPy, PyArrow, scikit-learn, LightGBM, RapidFuzz, PyYAML, and psutil

Do not run multiple retrieval or pipeline processes simultaneously.

For limited RAM, reduce workers, index batch size, retrieval chunk size, and
possibly `--max-df`. Lower retrieval settings may reduce candidate recall.

## Current Limitations

- Training candidate recall is approximately 0.45%, so retrieval is the main
  quality bottleneck.
- Full test retrieval requires large indexes and may exceed available RAM.
- Test output files are not complete until both TSV files pass the official
  validator.

## Completion Checklist

```powershell
Get-Process python
Get-ChildItem models
Get-Content reports/model_comparison.json
Get-Content models/final_config.json
Get-ChildItem output
Get-Content reports/pipeline_summary.json
```

Training is complete when the model files and `model_comparison.json` exist.
The complete project is finished only when `matching_results.tsv`,
`candidate_pairs.tsv`, and a passing `pipeline_summary.json` exist.
