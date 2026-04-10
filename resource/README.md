# Resource Directory

`resource/` is intentionally treated as local-only workspace data.

Put the following kinds of files here before running experiments:

- cleaned trajectory datasets, for example:
  `resource/...` if you want local copies, or any absolute path referenced by `default.yaml`
- generated query splits, such as `resource/queries` and `resource/queries_chengdu`
- shared similarity matrices under `resource/shared/similarity`
- experiment outputs under `resource/experiments`

Recommended workflow:

1. Copy or point `default.yaml` to your local dataset and query paths.
2. Generate or place query files for each dataset profile.
3. Run preprocessing or experiments, letting outputs stay under `resource/`.

This directory is ignored by Git except for this README.
