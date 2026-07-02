Models README
=============

This folder is intended to describe how to obtain or regenerate model artifacts
for this project. Do NOT commit binary model weights to the Git repository.

Recommended workflows
---------------------
1. Host final model weights on the Hugging Face Hub and reference the model
   identifier here. Example: `your-org/amazon-polarity-bert-finetuned`.
2. Provide a script or instructions to download/restore the model locally:

   ```bash
   # example: using transformers to download from HF Hub
   python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('your-org/your-model')"
   ```

3. If you prefer to store weights alongside the repo, use Git LFS and track
   large binary files (e.g., `*.safetensors`, `*.pt`). Note: Git LFS has
   bandwidth/storage implications; HF Hub is recommended for transformer
   models.

Files that belong here
----------------------
- `README.md` (this file): explains where to obtain model artifacts.
- `scripts/` (optional): helper scripts for downloading or packaging model
  artifacts.

Do not commit
-------------
- `model.safetensors`, `pytorch_model.bin`, optimizer checkpoints, and any
  other binary weight files — these should be excluded by `.gitignore`.
