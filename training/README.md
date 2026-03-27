# training/

Distillation training pipeline for llmem's two custom embedding models.

## Models

1. **Code Embedding Model** — 6-layer transformer, 384-dim output, ~22M params. Specialized for source code retrieval across Rust, Python, JS/TS, Go.

2. **Memory Embedding Model** — Same base architecture + temporal projection head. Specialized for llmem memory files with temporal awareness (recency, access frequency, memory type).

## Pipeline

```
1. Generate data    →  training/prompts/ + src/datagen/generate.py
2. Train models     →  scripts/train_code_model.py, scripts/train_memory_model.py
3. Export to ONNX   →  src/export/onnx_export.py
4. Deploy           →  dataset/models/*.onnx → used by crates/llmem-embed/
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Generate training data (requires LLM API access)
python scripts/generate_data.py --model claude --output ../dataset/

# Train code embedding model
python scripts/train_code_model.py --data ../dataset/code/generated/ --epochs 20

# Train memory embedding model
python scripts/train_memory_model.py --data ../dataset/memory/generated/ --epochs 20

# Export to ONNX
python -m src.export.onnx_export --checkpoint checkpoints/code_latest.pt --output ../dataset/models/code_embed_v1.onnx
python -m src.export.onnx_export --checkpoint checkpoints/memory_latest.pt --output ../dataset/models/memory_embed_v1.onnx
```

## Reference

- TurboQuant paper: `../dataset/turboquant.pdf`
- Quantization implementation: `../crates/llmem-quant/`
- ONNX inference in Rust: `../crates/llmem-embed/` (future)
