### Distance Functions

| Function | 32-d | 128-d | 384-d |
|---|---|---|---|
| `cosine_similarity` | 12 ns | 59 ns | 207 ns |
| `dot_product` | 4 ns | 28 ns | 120 ns |
| `l2_distance_squared` | 5 ns | 30 ns | 125 ns |
| `normalize` | 18 ns | 82 ns | 239 ns |

### HNSW Index (500 vectors, dim=32)

| Operation | Time |
|---|---|
| Build (500 inserts) | 32.7 ms |
| Search top-1 | 15.2 µs |
| Search top-10 | 15.2 µs |
| Search top-50 | 15.2 µs |
| Save to disk | 91 µs |
| Load from disk | 85 µs |

### IVF-Flat Index (500 vectors, dim=32)

| Operation | Time |
|---|---|
| Train (k-means, 16 clusters) | 2.2 ms |
| Search top-1 | 11.9 µs |
| Search top-10 | 12.0 µs |
| Search top-50 | 12.1 µs |
| Save to disk | 66 µs |
| Load from disk | 57 µs |

### TurboQuant MSE (dim=128)

| Bit-width | Quantize | Dequantize |
|---|---|---|
| 1-bit | 3.9 µs | 991 ns |
| 2-bit | 3.9 µs | 988 ns |
| 3-bit | 3.9 µs | 997 ns |
| 4-bit | 4.1 µs | 998 ns |

### TurboQuant Prod (dim=128)

| Bit-width | Quantize | Dequantize | IP Estimate |
|---|---|---|---|
| 2-bit | 116 µs | 141 µs | 111 µs |
| 3-bit | 115 µs | 111 µs | 111 µs |
| 4-bit | 115 µs | 111 µs | 112 µs |

### Bit Packing

| Operation | 128x2b | 384x2b | 384x4b |
|---|---|---|---|
| Pack | 161 ns | 539 ns | 264 ns |
| Unpack | 90 ns | 270 ns | 241 ns |

### Embedding Store

| Operation | 128d x 100 | 384d x 100 | 384d x 500 |
|---|---|---|---|
| `upsert` | TBD | TBD | TBD |
| `get` | TBD | TBD | TBD |
| `remove` | TBD | TBD | TBD |
| `save` | TBD | TBD | TBD |
| `load` | TBD | TBD | TBD |

### Inbox

| Operation | cap=7 | cap=50 |
|---|---|---|
| `push_to_capacity` | TBD | TBD |
| `push_with_eviction` | TBD | TBD |
| `save` | TBD | TBD |
| `load` | TBD | TBD |
| `drain` | TBD | TBD |

### Memory Index

| Operation | 10 entries | 100 entries |
|---|---|---|
| `parse_line` | TBD | — |
| `to_line` | TBD | — |
| `search` | TBD | TBD |
| `upsert_new` | TBD | TBD |
| `upsert_existing` | TBD | TBD |

### Eval Functions

| Function | 32d x 50 | 128d x 50 | 384d x 20 |
|---|---|---|---|
| `anisotropy` | TBD | TBD | TBD |
| `similarity_range` | TBD | TBD | TBD |
| `mean_center` | TBD | TBD | TBD |
| `discrimination_gap` | TBD | — | — |

> Measured on Apple Silicon (M-series) with `cargo bench`. Run `just bench` to reproduce.
