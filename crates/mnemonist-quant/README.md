# mnemonist-quant

TurboQuant vector quantization for mnemonist — near-optimal MSE and inner-product quantizers.

Implements the algorithms from [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874):

- **`TurboQuantMse`** — MSE-optimal quantizer using random rotation + Lloyd-Max codebooks
- **`TurboQuantProd`** — unbiased inner-product quantizer (MSE + QJL residual)
- **`CompressedEmbeddingStore`** — binary storage format for quantized embeddings

## Usage

```rust
use mnemonist_quant::{TurboQuantMse, TurboQuantProd, CompressedEmbeddingStore};
```

## References

- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research blog
- [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — arXiv:2504.19874
- [Optimal Quantization for Matrix Multiplication](https://arxiv.org/abs/2502.02617) — arXiv:2502.02617
- [Quantization of Large Language Models with an Overdetermined Linear System](https://arxiv.org/abs/2406.03482) — arXiv:2406.03482

## License

See [LICENSE](../../LICENSE) in the repository root.
