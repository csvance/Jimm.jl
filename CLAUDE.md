# Working in this repo

## After any non-trivial change, check these two files

1. **`README.md`** — the README enumerates supported backbones, families, and variant counts. After adding, removing, or renaming a model family, variant, or top-level public API, re-read README.md and update any list, table, or example that drifted. A change is "non-trivial" if a reader looking at README.md would now see something that is no longer true.

2. **`ci/jimm_ci/path_filter.py`** — this file maps changed source paths to the set of test families that CI runs. After adding a new model family, a new shared module under `src/`, a new top-level config file, or a new `test/test_*.jl` parity test, update `_FAMILY_PREFIXES`, `_FAMILY_EXACT`, `ALL_FAMILIES`, `REPRESENTATIVE_VARIANT`, and `_SHARED_PREFIXES` / `_SHARED_EXACT` as appropriate. If the filter is not updated, CI will silently skip tests for the new code on PR-scope runs; only a full `jimm-ci-run` master sweep would catch it.

When in doubt, run through this checklist explicitly before reporting a task complete.
