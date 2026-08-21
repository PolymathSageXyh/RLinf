# LeRobot dataset quality checker

`quality_check_lerobot_dataset.py` audits a local LeRobot v2 parquet dataset
without importing LeRobot or torch. It checks every frame and generates both
machine-readable metrics and deterministic sample visualizations.

Run it against a LeRobot dataset root, `collected_data` directory, or complete
collection run directory:

```bash
python toolkits/lerobot/quality_check_lerobot_dataset.py \
  --dataset-path logs/20260813-09:10:28 \
  --output-dir quality_reports/20260813-09-10-28
```

The output contains:

- `quality_report.md`: concise human-readable result and episode table.
- `quality_report.json`: full metadata, thresholds, issues, and per-episode metrics.
- `episode_metrics.csv`: one summary row per episode.
- `overview.png` and `signal_heatmap.png`: dataset-level visualizations.
- `samples/`: sampled frames and signal plots from representative or flagged episodes.

The checker validates parquet readability, schemas, shapes, null/NaN/Inf values,
metadata counts, indices, timestamps, terminal flags, action bounds, and image
decoding/quality. When the collection run also contains `demos/`, it compares
the replay-buffer index with the parquet episodes. When `run_embodiment.log` is
available, it correlates saved episodes with robot reflex and network warnings.

Results are classified as:

- `PASS`: no threshold violations.
- `REVIEW`: structurally valid data with warnings requiring human review.
- `FAIL`: a structural or data-integrity error.

By default a non-`PASS` result is reported but exits with status zero. Add
`--strict` for CI or scripted gates. Thresholds for normalized action bounds can
be changed with `--action-limit` and `--action-tolerance`; all effective
thresholds are stored in the JSON and Markdown reports.

