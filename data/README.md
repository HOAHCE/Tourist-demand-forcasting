# Data

Cleaned daily visitor series for four Hue heritage sites, used as model inputs.

| File | Description |
|---|---|
| `hue_dai_noi_daily_visitors_clean.csv` | Imperial City (Đại Nội) daily visitors |
| `hue_lang_khai_dinh_daily_visitors_clean.csv` | Khai Dinh Tomb daily visitors |
| `hue_lang_minh_mang_daily_visitors_clean.csv` | Minh Mang Tomb daily visitors |
| `hue_lang_tu_duc_daily_visitors_clean.csv` | Tu Duc Tomb daily visitors |
| `hue_four_destinations_split_summary.csv` | Per-site date range, split sizes, zero-day and total counts |
| `hue_four_destinations_share_summary.csv` | Each site's share of domestic/international/total visitors |

Each visitor file spans 1,393 consecutive days (2022-11-01 → 2026-08-24); the experiment aligns to a common start of 2022-11-10 (1,384 days) and splits chronologically into 966 train / 208 validation / 210 test days.

Columns: `date`, `destination_id`, `destination`, `domestic_visitors`, `international_visitors`, `visitors` (target = domestic + international), `log1p_visitors`, `split`, `is_zero_day`, `data_quality_flag`.

SHA-256 checksums of the four visitor files are recorded in `../results/tables/data_audit.csv` and `../results/logs/run_manifest.json`.

Source: compiled from public pages of the Hue Monuments Conservation Centre.
