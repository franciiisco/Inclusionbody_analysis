# APIC — Agent Guide

## Quick start

```bash
pip install -r requirements.txt
python run.py              # GUI
python main.py             # CLI single image (default: data/raw/sample_image.png)
python main.py --batch --input "data/raw" --pattern "*.tif"  # batch
```

## Entrypoints

| File | Purpose |
|------|---------|
| `run.py` | GUI launcher (ttkbootstrap/tkinter). Import: `from app.gui import main` |
| `main.py` | CLI using argparse. Calls `src.core.{process_image_v2,batch_process}` |
| `APIC.spec` | PyInstaller build for standalone .exe (entry: `app/gui.py`) |

## Project layout

```
src/               # Image processing pipeline
  core.py          # Orchestration: process_image_v2, batch_process_v2
  preprocessing.py # CLAHE normalization, bilateral denoise, illumination correction
  segmentation.py  # Watershed cell segmentation, Otsu/adaptive thresholding
  detection_v2_0.py# Active inclusion detection algorithm (v2.0; v1 is legacy)
  analysis.py      # Statistics, filename metadata extraction, Excel export
  features.py      # Morphological feature extraction for cells/inclusions
  visualization.py # Result overlays with color-coded inclusions
app/               # GUI components
  gui.py           # Main window class + main()
  components/      # ProgressTracker, ResultsViewer
  tabs/            # analysis_tab, methodology_tab, info_tab
utils/             # file_operations.py, visualization.py
tests/
  filename_test.py # Only test — metadata extraction from filenames
config.py          # All parameters in one place
data/raw/          # Input images
data/processed/    # Output: Imagenes/ + Datos_crudos/
```

- All processing lives in `src/`. The v2.0 algorithm (`detection_v2_0.py`) is the active one; v1 is legacy/dead code.
- `config.py` is the single source of truth for pipeline parameters. `DEVELOPMENT_MODE = True` enables interactive matplotlib visualizations.

## Tests

```bash
python -m unittest tests.filename_test -v
```

One test file, `unittest`-based. No test runner config (no pytest, no tox). No CI.

## `src/analysis.py` — key functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `extract_metadata_from_filename(filename)` | `dict \| None` | Parses `CONDICION_BOTE_REPLICA_TIEMPO_NUMERO` flexibly; missing fields get defaults (`B1`/`R1`/`001`/`t0`), returns `None` if < 3 markers found |
| `validate_filename_format(filename)` | `bool` | Wraps extraction — checks that condicion + replica + tiempo are present |
| `summarize_inclusions(inclusions, segments)` | `dict` | Per-image stats: total cells, cells with inclusions, inclusion count, areas, ratios |
| `aggregate_inclusion_data(image_results)` | `dict` | Groups results by `condition/time` and `condition/time/replicate`; computes mean/std across images per group |
| `export_results_to_excel(image_results, output_dir)` | `str` | Writes per-replica sheets + `Promedios_Generales` sheet to `resultados_analisis_polifosfatos.xlsx`; falls back with timestamped name if file is locked |

## File naming convention

Batch processing validates filenames by default: `CONDICION_BOTE_REPLICA_TIEMPO_NUMERO.ext`. Metadata extraction (`extract_metadata_from_filename`) is flexible — missing fields get defaults. Pass `--no-enforce-naming` to skip validation.

## Known quirks

- `sys.path.insert(0, ...)` used in `run.py`, `tests/filename_test.py`, and `app/gui.py`. Always run from repo root.
- No formatter, linter, or type checker configured.
- Output subdirectories are created as `Imagenes/` and `Datos_crudos/` under the output dir.
- Batch default pattern differs: `*.png` in CLI, `*.tif` in GUI.
- The `progress_callback` parameter on `batch_process` is used by the GUI for progress tracking; no-op in CLI.
- `--backend omnipose` has been removed. Segmentation uses the classical watershed pipeline.
