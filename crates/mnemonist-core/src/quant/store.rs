//! Compressed embedding store using TurboQuant quantization.
//!
//! Binary format (LMCQ):
//! ```text
//! Header: [magic: 4B "LMCQ"] [version: u8] [dimension: u32] [count: u32]
//!         [bits: u8] [quant_type: u8 (0=mse, 1=prod)] [rotation_seed: u64]
//!         [qjl_seed: u64 (only if prod)]
//! Entry:  [file_len: u16] [file: bytes] [hash: u64]
//!         [norm: f32] [packed_indices: ceil(dim * bits_mse / 8) bytes]
//!         (if prod: [residual_norm: f32] [qjl_bits: ceil(dim / 8) bytes])
//! ```

use std::fs;
use std::io::{Read as _, Write as _};
use std::path::Path;

use super::QuantError;
use super::pack;

const MAGIC: &[u8; 4] = b"LMCQ";
const FORMAT_VERSION: u8 = 1;

/// Type of quantization used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuantType {
    Mse = 0,
    Prod = 1,
}

/// A single compressed embedding entry.
#[derive(Debug, Clone)]
pub struct CompressedEntry {
    pub file: String,
    pub hash: u64,
    pub norm: f32,
    pub packed_indices: Vec<u8>,
    /// Only present for QuantType::Prod.
    pub residual_norm: Option<f32>,
    /// Only present for QuantType::Prod.
    pub qjl_bits: Option<Vec<u8>>,
}

/// Compressed embedding store using TurboQuant.
#[derive(Debug, Clone)]
pub struct CompressedEmbeddingStore {
    pub dimension: usize,
    pub bits: u8,
    pub quant_type: QuantType,
    pub rotation_seed: u64,
    pub qjl_seed: Option<u64>,
    pub entries: Vec<CompressedEntry>,
}

impl CompressedEmbeddingStore {
    /// Create a new empty store.
    pub fn new(
        dimension: usize,
        bits: u8,
        quant_type: QuantType,
        rotation_seed: u64,
        qjl_seed: Option<u64>,
    ) -> Self {
        Self {
            dimension,
            bits,
            quant_type,
            rotation_seed,
            qjl_seed,
            entries: Vec::new(),
        }
    }

    /// The effective MSE bit-width (bits for MSE, bits-1 for Prod).
    fn mse_bits(&self) -> u8 {
        match self.quant_type {
            QuantType::Mse => self.bits,
            QuantType::Prod => self.bits - 1,
        }
    }

    /// Save to a binary file.
    pub fn save(&self, path: &Path) -> Result<(), QuantError> {
        let mut buf = Vec::new();

        // Header
        buf.write_all(MAGIC)?;
        buf.write_all(&[FORMAT_VERSION])?;
        buf.write_all(&(self.dimension as u32).to_le_bytes())?;
        buf.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        buf.write_all(&[self.bits])?;
        buf.write_all(&[self.quant_type as u8])?;
        buf.write_all(&self.rotation_seed.to_le_bytes())?;
        if self.quant_type == QuantType::Prod {
            buf.write_all(&self.qjl_seed.unwrap_or(0).to_le_bytes())?;
        }

        let mse_bits = self.mse_bits();
        let indices_size = pack::packed_byte_size(self.dimension, mse_bits);
        let qjl_size = self.dimension.div_ceil(8);

        // Entries
        for entry in &self.entries {
            let file_bytes = entry.file.as_bytes();
            buf.write_all(&(file_bytes.len() as u16).to_le_bytes())?;
            buf.write_all(file_bytes)?;
            buf.write_all(&entry.hash.to_le_bytes())?;
            buf.write_all(&entry.norm.to_le_bytes())?;

            assert_eq!(entry.packed_indices.len(), indices_size);
            buf.write_all(&entry.packed_indices)?;

            if self.quant_type == QuantType::Prod {
                let rn = entry.residual_norm.unwrap_or(0.0);
                buf.write_all(&rn.to_le_bytes())?;

                let default_qjl = vec![0u8; qjl_size];
                let qjl = entry.qjl_bits.as_deref().unwrap_or(&default_qjl);
                assert_eq!(qjl.len(), qjl_size);
                buf.write_all(qjl)?;
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, buf)?;
        Ok(())
    }

    /// Load from a binary file.
    pub fn load(path: &Path) -> Result<Self, QuantError> {
        let data = fs::read(path)?;
        let mut cursor = &data[..];

        // Magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(QuantError::Format("invalid magic bytes".into()));
        }

        // Version
        let mut ver = [0u8; 1];
        cursor.read_exact(&mut ver)?;
        if ver[0] != FORMAT_VERSION {
            return Err(QuantError::Format(format!(
                "unsupported version: {}",
                ver[0]
            )));
        }

        // Dimension
        let mut dim_bytes = [0u8; 4];
        cursor.read_exact(&mut dim_bytes)?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        // Count
        let mut count_bytes = [0u8; 4];
        cursor.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        // Bits
        let mut bits_byte = [0u8; 1];
        cursor.read_exact(&mut bits_byte)?;
        let bits = bits_byte[0];

        // Quant type
        let mut qt_byte = [0u8; 1];
        cursor.read_exact(&mut qt_byte)?;
        let quant_type = match qt_byte[0] {
            0 => QuantType::Mse,
            1 => QuantType::Prod,
            v => return Err(QuantError::Format(format!("unknown quant type: {v}"))),
        };

        // Rotation seed
        let mut seed_bytes = [0u8; 8];
        cursor.read_exact(&mut seed_bytes)?;
        let rotation_seed = u64::from_le_bytes(seed_bytes);

        // QJL seed (only for prod)
        let qjl_seed = if quant_type == QuantType::Prod {
            let mut qjl_seed_bytes = [0u8; 8];
            cursor.read_exact(&mut qjl_seed_bytes)?;
            Some(u64::from_le_bytes(qjl_seed_bytes))
        } else {
            None
        };

        let mse_bits = match quant_type {
            QuantType::Mse => bits,
            QuantType::Prod => bits - 1,
        };
        let indices_size = pack::packed_byte_size(dimension, mse_bits);
        let qjl_size = dimension.div_ceil(8);

        // Entries
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            // File name
            let mut file_len_bytes = [0u8; 2];
            cursor.read_exact(&mut file_len_bytes)?;
            let file_len = u16::from_le_bytes(file_len_bytes) as usize;
            let mut file_bytes = vec![0u8; file_len];
            cursor.read_exact(&mut file_bytes)?;
            let file = String::from_utf8(file_bytes)
                .map_err(|e| QuantError::Format(format!("invalid UTF-8: {e}")))?;

            // Hash
            let mut hash_bytes = [0u8; 8];
            cursor.read_exact(&mut hash_bytes)?;
            let hash = u64::from_le_bytes(hash_bytes);

            // Norm
            let mut norm_bytes = [0u8; 4];
            cursor.read_exact(&mut norm_bytes)?;
            let norm = f32::from_le_bytes(norm_bytes);

            // Packed indices
            let mut packed_indices = vec![0u8; indices_size];
            cursor.read_exact(&mut packed_indices)?;

            // Prod-specific fields
            let (residual_norm, qjl_bits) = if quant_type == QuantType::Prod {
                let mut rn_bytes = [0u8; 4];
                cursor.read_exact(&mut rn_bytes)?;
                let rn = f32::from_le_bytes(rn_bytes);

                let mut qjl = vec![0u8; qjl_size];
                cursor.read_exact(&mut qjl)?;

                (Some(rn), Some(qjl))
            } else {
                (None, None)
            };

            entries.push(CompressedEntry {
                file,
                hash,
                norm,
                packed_indices,
                residual_norm,
                qjl_bits,
            });
        }

        Ok(Self {
            dimension,
            bits,
            quant_type,
            rotation_seed,
            qjl_seed,
            entries,
        })
    }

    /// Find an entry by filename.
    pub fn get(&self, file: &str) -> Option<&CompressedEntry> {
        self.entries.iter().find(|e| e.file == file)
    }

    /// Insert or update an entry.
    pub fn upsert(&mut self, entry: CompressedEntry) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.file == entry.file) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Remove an entry by filename.
    pub fn remove(&mut self, file: &str) -> bool {
        let len = self.entries.len();
        self.entries.retain(|e| e.file != file);
        self.entries.len() < len
    }

    /// Storage size in bytes for the entry data (excluding header).
    pub fn data_size(&self) -> usize {
        let mse_bits = self.mse_bits();
        let indices_size = pack::packed_byte_size(self.dimension, mse_bits);
        let per_entry = 2 + 8 + 4 + indices_size // file_len + hash + norm + indices
            + if self.quant_type == QuantType::Prod {
                4 + self.dimension.div_ceil(8) // residual_norm + qjl_bits
            } else {
                0
            };
        self.entries.len() * per_entry
    }

    /// Equivalent uncompressed size (f32 embeddings) for comparison.
    pub fn uncompressed_size(&self) -> usize {
        self.entries.len() * self.dimension * 4
    }

    /// Compression ratio (uncompressed / compressed).
    pub fn compression_ratio(&self) -> f32 {
        let compressed = self.data_size();
        if compressed == 0 {
            return 0.0;
        }
        self.uncompressed_size() as f32 / compressed as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_store(quant_type: QuantType) -> CompressedEmbeddingStore {
        let dim = 64;
        let bits: u8 = 2;
        let mse_bits = match quant_type {
            QuantType::Mse => bits,
            QuantType::Prod => bits - 1,
        };
        let indices_size = pack::packed_byte_size(dim, mse_bits);
        let qjl_size = (dim + 7) / 8;

        let mut store = CompressedEmbeddingStore::new(
            dim,
            bits,
            quant_type,
            42,
            if quant_type == QuantType::Prod {
                Some(99)
            } else {
                None
            },
        );

        store.upsert(CompressedEntry {
            file: "feedback_test.md".into(),
            hash: 12345,
            norm: 1.5,
            packed_indices: vec![0xAB; indices_size],
            residual_norm: if quant_type == QuantType::Prod {
                Some(0.3)
            } else {
                None
            },
            qjl_bits: if quant_type == QuantType::Prod {
                Some(vec![0xCD; qjl_size])
            } else {
                None
            },
        });

        store.upsert(CompressedEntry {
            file: "user_prefs.md".into(),
            hash: 67890,
            norm: 2.0,
            packed_indices: vec![0x12; indices_size],
            residual_norm: if quant_type == QuantType::Prod {
                Some(0.1)
            } else {
                None
            },
            qjl_bits: if quant_type == QuantType::Prod {
                Some(vec![0x34; qjl_size])
            } else {
                None
            },
        });

        store
    }

    #[test]
    fn mse_store_roundtrip() {
        let store = make_test_store(QuantType::Mse);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.lmcq");

        store.save(&path).unwrap();
        let loaded = CompressedEmbeddingStore::load(&path).unwrap();

        assert_eq!(loaded.dimension, store.dimension);
        assert_eq!(loaded.bits, store.bits);
        assert_eq!(loaded.quant_type, QuantType::Mse);
        assert_eq!(loaded.rotation_seed, 42);
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries[0].file, "feedback_test.md");
        assert_eq!(loaded.entries[0].norm, 1.5);
        assert_eq!(
            loaded.entries[0].packed_indices,
            store.entries[0].packed_indices
        );
    }

    #[test]
    fn prod_store_roundtrip() {
        let store = make_test_store(QuantType::Prod);
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.lmcq");

        store.save(&path).unwrap();
        let loaded = CompressedEmbeddingStore::load(&path).unwrap();

        assert_eq!(loaded.quant_type, QuantType::Prod);
        assert_eq!(loaded.qjl_seed, Some(99));
        assert_eq!(loaded.entries[0].residual_norm, Some(0.3));
        assert!(loaded.entries[0].qjl_bits.is_some());
    }

    #[test]
    fn compression_ratio_positive() {
        let store = make_test_store(QuantType::Mse);
        let ratio = store.compression_ratio();
        assert!(ratio > 1.0, "compression ratio should be > 1, got {ratio}");
    }

    #[test]
    fn upsert_replaces() {
        let mut store = make_test_store(QuantType::Mse);
        let old_hash = store.entries[0].hash;

        store.upsert(CompressedEntry {
            file: "feedback_test.md".into(),
            hash: 99999,
            norm: 3.0,
            packed_indices: store.entries[0].packed_indices.clone(),
            residual_norm: None,
            qjl_bits: None,
        });

        assert_eq!(store.entries.len(), 2);
        assert_ne!(store.get("feedback_test.md").unwrap().hash, old_hash);
    }

    #[test]
    fn remove_entry() {
        let mut store = make_test_store(QuantType::Mse);
        assert!(store.remove("feedback_test.md"));
        assert_eq!(store.entries.len(), 1);
        assert!(!store.remove("nonexistent.md"));
    }
}
