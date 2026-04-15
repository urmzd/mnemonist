//! Bit-packing utilities for storing b-bit indices compactly.
//!
//! For b-bit quantization of d dimensions, we pack indices into
//! ceil(d * b / 8) bytes.

use super::QuantError;

/// Pack an array of b-bit indices into a byte vector.
///
/// Each `index` value must fit in `bits` bits (i.e., < 2^bits).
pub fn pack_indices(indices: &[u8], bits: u8) -> Result<Vec<u8>, QuantError> {
    if bits == 0 || bits > 4 {
        return Err(QuantError::UnsupportedBitWidth(bits));
    }

    let total_bits = indices.len() * bits as usize;
    let byte_len = total_bits.div_ceil(8);
    let mut packed = vec![0u8; byte_len];

    let mut bit_offset = 0usize;
    for &idx in indices {
        debug_assert!(idx < (1 << bits), "index {idx} exceeds {bits}-bit range");
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        // Write the bits. May span two bytes if bits cross a byte boundary.
        packed[byte_pos] |= idx << bit_pos;
        if bit_pos + bits as usize > 8 {
            packed[byte_pos + 1] |= idx >> (8 - bit_pos);
        }

        bit_offset += bits as usize;
    }

    Ok(packed)
}

/// Unpack b-bit indices from a packed byte vector.
///
/// Returns exactly `count` indices.
pub fn unpack_indices(packed: &[u8], bits: u8, count: usize) -> Result<Vec<u8>, QuantError> {
    if bits == 0 || bits > 4 {
        return Err(QuantError::UnsupportedBitWidth(bits));
    }

    let mask = (1u8 << bits) - 1;
    let mut indices = Vec::with_capacity(count);

    let mut bit_offset = 0usize;
    for _ in 0..count {
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        let mut val = packed[byte_pos] >> bit_pos;
        if bit_pos + bits as usize > 8 {
            val |= packed[byte_pos + 1] << (8 - bit_pos);
        }
        indices.push(val & mask);

        bit_offset += bits as usize;
    }

    Ok(indices)
}

/// Calculate the packed byte size for `count` indices at `bits` per index.
pub fn packed_byte_size(count: usize, bits: u8) -> usize {
    (count * bits as usize).div_ceil(8)
}

/// Pack sign bits (±1 values) into bytes. Each bit: 0 = negative, 1 = positive.
pub fn pack_signs(signs: &[i8]) -> Vec<u8> {
    let byte_len = signs.len().div_ceil(8);
    let mut packed = vec![0u8; byte_len];

    for (i, &s) in signs.iter().enumerate() {
        if s > 0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }

    packed
}

/// Unpack sign bits from packed bytes. Returns ±1 values.
pub fn unpack_signs(packed: &[u8], count: usize) -> Vec<i8> {
    let mut signs = Vec::with_capacity(count);
    for i in 0..count {
        let bit = (packed[i / 8] >> (i % 8)) & 1;
        signs.push(if bit == 1 { 1 } else { -1 });
    }
    signs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_1bit() {
        let indices: Vec<u8> = vec![0, 1, 1, 0, 1, 0, 0, 1];
        let packed = pack_indices(&indices, 1).unwrap();
        let unpacked = unpack_indices(&packed, 1, indices.len()).unwrap();
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_2bit() {
        let indices: Vec<u8> = vec![0, 1, 2, 3, 3, 2, 1, 0];
        let packed = pack_indices(&indices, 2).unwrap();
        assert_eq!(packed.len(), 2); // 8 * 2 bits = 16 bits = 2 bytes
        let unpacked = unpack_indices(&packed, 2, indices.len()).unwrap();
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_3bit() {
        let indices: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let packed = pack_indices(&indices, 3).unwrap();
        assert_eq!(packed.len(), 3); // 8 * 3 bits = 24 bits = 3 bytes
        let unpacked = unpack_indices(&packed, 3, indices.len()).unwrap();
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_4bit() {
        let indices: Vec<u8> = vec![0, 5, 10, 15, 3, 7, 12, 1];
        let packed = pack_indices(&indices, 4).unwrap();
        assert_eq!(packed.len(), 4); // 8 * 4 bits = 32 bits = 4 bytes
        let unpacked = unpack_indices(&packed, 4, indices.len()).unwrap();
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_odd_count() {
        // Non-byte-aligned: 5 values at 3 bits = 15 bits → 2 bytes
        let indices: Vec<u8> = vec![7, 0, 3, 5, 2];
        let packed = pack_indices(&indices, 3).unwrap();
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_indices(&packed, 3, indices.len()).unwrap();
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn packed_byte_size_correct() {
        assert_eq!(packed_byte_size(8, 1), 1);
        assert_eq!(packed_byte_size(8, 2), 2);
        assert_eq!(packed_byte_size(8, 3), 3);
        assert_eq!(packed_byte_size(8, 4), 4);
        assert_eq!(packed_byte_size(5, 3), 2);
        assert_eq!(packed_byte_size(384, 2), 96);
    }

    #[test]
    fn sign_pack_unpack() {
        let signs: Vec<i8> = vec![1, -1, -1, 1, 1, 1, -1, -1, 1];
        let packed = pack_signs(&signs);
        let unpacked = unpack_signs(&packed, signs.len());
        assert_eq!(signs, unpacked);
    }
}
