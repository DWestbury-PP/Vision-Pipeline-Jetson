# Moondream Model Cache

This directory caches the Moondream Vision Language Model and its associated files.

## Contents

When the Moondream service runs, it will automatically download and cache:

- Model weights (`pytorch_model.bin` or similar)
- Tokenizer files (`tokenizer.json`, `vocab.json`, etc.)
- Configuration files (`config.json`)
- Any other model artifacts

## Default Model

- **Model**: `vikhyatk/moondream2`
- **Source**: Hugging Face Hub
- **Size**: ~1.6GB
- **Cache Location**: This directory

## Environment Variables

You can customize the model by setting:

```bash
MOONDREAM_MODEL=vikhyatk/moondream2  # Default model
```

## Note

Files in this directory are automatically managed by the Moondream service and should not be manually edited.
