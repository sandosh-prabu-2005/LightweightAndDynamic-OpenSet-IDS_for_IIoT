# IDS Inference Runner

This folder contains a standalone inference pipeline for the saved teacher checkpoints in `saved_models/`.

## Main Script

- `inference_pipeline.py`

## What It Supports

- Full-model `.pt` checkpoints
- `state_dict`-only `.pt` checkpoints
- Automatic CPU/GPU selection
- Manual single-sample inference
- CSV inference
- NumPy `.npy` / `.npz` inference
- DataFrame input through the reusable Python functions
- Softmax confidence
- Label decoding
- Batch inference
- Confusion matrix generation
- Accuracy / F1 evaluation
- Probability threshold tuning

## Important Preprocessing Note

The current repo includes the `.pt` model files, but it does not currently include the fitted preprocessing artifacts like:

- `*_feature_selector.pkl`
- `*_feature_scaler.pkl`
- `*_le.pkl`

If those files are added later beside the model, the script will automatically use them.

If those files are not present, the script expects numeric inputs that already match the checkpoint feature dimension:

- `NSL-KDD_teacher.pt` -> `20` features
- `CICIDS2017_teacher.pt` -> `20` features
- `Gas_Pipeline_teacher.pt` -> `20` features
- `Water_Storage_teacher.pt` -> `23` features

## Example Commands

```powershell
python load_model_run\inference_pipeline.py `
  --model saved_models\NSL-KDD_teacher.pt `
  --csv load_model_run\examples\nsl_kdd_dummy_input.csv `
  --label-column label `
  --evaluate `
  --output-dir load_model_run\outputs\nsl
```

```powershell
python load_model_run\inference_pipeline.py `
  --model saved_models\Water_Storage_teacher.pt `
  --manual-values "0.10,0.20,0.15,0.30,0.40,0.12,0.08,0.55,0.67,0.90,0.14,0.18,0.22,0.28,0.31,0.27,0.19,0.16,0.11,0.09,0.05,0.03,0.01"
```

```powershell
python load_model_run\inference_pipeline.py `
  --model saved_models\CICIDS2017_teacher.pt `
  --csv load_model_run\examples\cicids2017_dummy_input.csv `
  --label-column label `
  --evaluate `
  --tune-threshold `
  --print-logits `
  --results-csv load_model_run\outputs\cic_predictions.csv `
  --output-dir load_model_run\outputs\cic
```
