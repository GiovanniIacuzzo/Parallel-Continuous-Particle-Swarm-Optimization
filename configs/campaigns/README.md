# Campaign configurations

- `definitive_architecture_campaign.json`: predeclared confirmatory multi-stage
  campaign (9,360 optimization runs and 273,888,000 objective evaluations;
  30 search seeds and 10 timing seeds).
- `campaign_smoke.json`: integration-only campaign for checking orchestration,
  checkpointing and analysis. Never use its outputs for scientific conclusions.
- `cpso_*_pilot.json`: broader exploratory designs intended for development
  machines; their outputs are not confirmatory.
- `cpso_*_m2_pilot.json` and `cpso_dt_sensitivity_m2_screening.json`: local
  Apple-Silicon profiles. They retain the per-run FE budgets of the broader
  pilot designs and reduce replication count where needed. Runtime measurements
  are specific to the Mac and cannot be compared directly with HPC wall times.

The root `README.md` contains the build, campaign execution and analysis
workflow. Each JSON records the campaign's `evidence_class`; the runner carries
that designation into `campaign_manifest.json` and the analysis report.
