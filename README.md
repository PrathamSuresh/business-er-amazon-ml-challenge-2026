# Business Entity Resolution

## About this project

Built for the **Amazon ML Challenge 2026** (Business Entity Resolution) as a way to learn how
AWS and cloud ML workflows work in practice: storing data in the cloud, running ML workloads on
managed infrastructure, and building a reproducible pipeline end to end.

The task: given business records from three sources with noisy names and addresses
(abbreviations, typos, other scripts, missing fields), find which Source 2 / Source 3 records
refer to the same real-world business as each Source 1 record. Scored with macro F0.5.

**Result:** public leaderboard F0.5 = **0.900** (v5), up from a validation F0.5 of 0.796 (v4),
mainly by adding address-based blocking keys (blocking recall 0.70 → ~0.96).

## Infrastructure used

- **Amazon S3**: private bucket (public access blocked, SSE-S3 encryption) holding the
  challenge dataset.
- **Amazon SageMaker AI (Studio, JupyterLab)**: data exploration (`scripts/eda.py`) and the
  first pipeline runs, copying data from S3.
- **Service Quotas**: enabling Studio apps and requesting larger instance types on a new account.
- **Kaggle Notebooks (CPU)**: the full-scale runs. The account's larger SageMaker instance quota
  was still pending, and the 4 GB `ml.t3.medium` instance could not hold ~24M records in memory,
  so the final pipeline ran on a larger CPU notebook. The code is identical on both; only the
  resource flags differ.

Only open-source libraries are used (DuckDB, LightGBM, pandas, anyascii). No external data,
APIs, geocoding or pretrained models, in line with the challenge rules.

## Data exploration findings

- ~2.2M / 1.7M Source 1 records and ~10M Source 2+3 records per split (train / test).
- Every true match is within the same country, so blocking is done per country.
- Every Source 2/3 record belongs to at most one Source 1 entity (7,638,365 links over 7,638,365
  distinct records), so each record is assigned to its single best-scoring candidate.
- ~26% of Source 2/3 records match nothing; only ~5.6% of Source 1 entities have no match.
- Names are heavily noisy (only ~11% exact matches): Indian scripts, websites instead of names,
  digit/letter swaps, dropped vowels, shuffled words; addresses are reordered and abbreviated.

## How it works

1. **Cleaning** (`src/clean_sql.py`, stage `prep`): transliterate non-Latin scripts with
   `anyascii`, lowercase, reduce website names to their domain (`gmlabs.com` → `gmlabs`), fix
   digit/letter swaps inside words (`regiona1` → `regional`), expand abbreviations
   (`pvt` → `private`), strip legal words to get a *core name*, normalise address abbreviations
   and US state names, extract house numbers and postal codes. Country-agnostic, so France
   (absent from training) is handled the same way.
2. **Blocking** (stage `block`): each record gets hashed keys, all scoped to its country:
   - name keys: rare words, consonant skeletons (first letter + consonants, doubles squeezed:
     `limited`/`limittedd` → `lmtd`), the whole sorted core name, adjacent word pairs;
   - address keys: rare address words, house number + address word, first name word + house
     number.

   Keys shared by more than `--cap` Source 1 records are ignored; each Source 2/3 record uses
   its `--nkeys` rarest keys. The `--topk` best candidates per Source 2/3 record (ranked by a
   cheap name + address similarity) are kept and written to `candidate_pairs.tsv`.
3. **Features** (stage `feat`): Jaro-Winkler / Levenshtein similarity of names (raw, core,
   word-sorted, skeleton), word overlap, address similarity and overlap, house-number and
   postal-code agreement, number of shared name/address keys, and relative features (how a
   candidate compares with the other candidates of the same record / entity).
4. **Model** (stage `train`): LightGBM classifier (gradient-boosted trees, MIT licence, trained
   from scratch) on candidate pairs of a sample of Source 1 entities, labelled from the ground
   truth. Each Source 2/3 record is assigned only to its highest-scoring candidate, and only if
   that score passes a threshold tuned for macro F0.5 on held-out Source 1 entities.
5. **Output** (stage `predict`): `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Reproduce

```bash
pip install -r requirements.txt
python src/pipeline.py --data-dir <path>/dataset \
    --memory 20GB --threads 4 --buckets 8 --fit-permille 100
```

This is the configuration used for the submitted results (a machine with ~30 GB RAM, 4 CPUs;
about 1.5–2 hours). On a small machine, lower `--memory` / `--threads` and raise `--buckets`.
Intermediate results are stored in a DuckDB file (`--work-dir`, default `~/er_work`), so stages
can be rerun individually, e.g. `--stages train,predict`. Quick end-to-end test on a small
slice: `--smoke-rows 50000 --work-dir ~/er_smoke`.

The log reports blocking recall (overall and by number of candidates kept) and the validation
macro F0.5. Validate the output format with the organisers' script:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
