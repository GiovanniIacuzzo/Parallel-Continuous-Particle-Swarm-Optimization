# Active experiment configurations

**Current stage (dt, do not run both unless intended):**

- `cpso_dt_sweep_100_m2.json`: historical exploratory 100-point grid; preserve for provenance; no need to rerun.
- `cpso_dt_statistical_selection_m2.json`: independent statistical selection study (30 paired seeds); active candidate for next campaign.
- `campaign_smoke.json`: tiny integration-only sanity check.

Do not treat older `definitive_architecture_campaign.json` (archived) as final: its dt value has not been locked.
Historic configurations are provided in the separate recovery ZIP. Fixtures in `tests/fixtures/configs/` are retained only to keep regression checks self-contained.
