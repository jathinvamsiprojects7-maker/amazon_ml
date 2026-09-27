# Amazon ML Challenge 2026

Business Entity Resolution project for the Amazon ML Challenge 2026.

## Current Status

The project is partially completed.

### Completed

- Dataset analysis completed.
- Data normalization/preprocessing pipeline created.
- Retrieval/indexing code implemented.
- S2 candidate generation was completed.
- S3 candidate generation was started.
- S3 candidate generation was stopped before completion was verified.
- Architecture and retrieval experiments are documented in `reports/`.
- Project source code is in `src/`.

### Important

The raw datasets, large cache files, models and outputs are NOT stored in this GitHub repository.

The teammate already has the official challenge dataset and can recreate the preprocessing on their machine.

## Current Checkpoint

The previous machine had:

- S1: ~2.2M records
- S2: ~5.0M records
- S3: ~5.3M records

S2 candidate generation produced about 10 GB of candidate data.

S3 candidate generation produced about 9 GB in 111 parquet parts before it was stopped.

S3 completion was NOT verified.

## What To Do Next

1. Clone this repository.
2. Put the official challenge dataset in the expected dataset location.
3. Run the existing normalization/preprocessing pipeline.
4. Run/continue S3 candidate generation.
5. Verify candidate recall against the training ground truth.
6. Continue with:
   - candidate union/deduplication
   - feature generation
   - training
   - validation
   - threshold tuning
   - test inference
   - final output generation
   - official submission validation

## Important Rule

Do not assume a stage is complete just because files exist.

Check the logs, row counts, metrics and outputs before moving to the next stage.

Do not redesign the project unnecessarily. Continue from the existing architecture in `FINAL_ARCHITECTURE.txt`.

## For AI Coding Agents

First inspect:

- `FINAL_ARCHITECTURE.txt`
- `src/`
- `run_normalize.py`
- `run_retrieval.py`
- `run_features.py`
- `run_pipeline.py`
- `reports/`

Then determine the exact current state before running anything.

The goal is to continue the existing project, not restart it.
