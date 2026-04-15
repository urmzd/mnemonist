use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mnemonist_core::quant::mse::TurboQuantMse;
use mnemonist_core::quant::pack;
use mnemonist_core::quant::prod::TurboQuantProd;

fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic pseudo-random vector using xorshift
    let mut state = seed;
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn bench_mse_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant_mse/quantize");

    for dim in [64, 128, 384] {
        let quant = TurboQuantMse::new(dim, 2, 42).unwrap();
        let x = random_unit_vector(dim, 7);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| quant.quantize(&x).unwrap());
        });
    }

    group.finish();
}

fn bench_mse_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant_mse/dequantize");

    for dim in [64, 128, 384] {
        let quant = TurboQuantMse::new(dim, 2, 42).unwrap();
        let x = random_unit_vector(dim, 7);
        let q = quant.quantize(&x).unwrap();

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| quant.dequantize(&q).unwrap());
        });
    }

    group.finish();
}

fn bench_mse_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant_mse/bits");
    let dim = 128;
    let x = random_unit_vector(dim, 7);

    for bits in 1..=4u8 {
        let quant = TurboQuantMse::new(dim, bits, 42).unwrap();

        group.bench_with_input(BenchmarkId::new("quantize", bits), &bits, |bench, _| {
            bench.iter(|| quant.quantize(&x).unwrap());
        });

        let q = quant.quantize(&x).unwrap();
        group.bench_with_input(BenchmarkId::new("dequantize", bits), &bits, |bench, _| {
            bench.iter(|| quant.dequantize(&q).unwrap());
        });
    }

    group.finish();
}

fn bench_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant_prod");
    let dim = 128;
    let x = random_unit_vector(dim, 7);
    let query = random_unit_vector(dim, 99);

    for bits in 2..=4u8 {
        let quant = TurboQuantProd::new(dim, bits, 42, 99).unwrap();

        group.bench_with_input(BenchmarkId::new("quantize", bits), &bits, |bench, _| {
            bench.iter(|| quant.quantize(&x).unwrap());
        });

        let q = quant.quantize(&x).unwrap();

        group.bench_with_input(BenchmarkId::new("dequantize", bits), &bits, |bench, _| {
            bench.iter(|| quant.dequantize(&q).unwrap());
        });

        group.bench_with_input(
            BenchmarkId::new("inner_product_estimate", bits),
            &bits,
            |bench, _| {
                bench.iter(|| quant.inner_product_estimate(&query, &q).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack");

    for (dim, bits) in [(128, 2u8), (384, 2), (384, 4)] {
        let indices: Vec<u8> = (0..dim).map(|i| (i % (1 << bits)) as u8).collect();

        group.bench_with_input(
            BenchmarkId::new("pack", format!("{dim}x{bits}b")),
            &dim,
            |bench, _| {
                bench.iter(|| pack::pack_indices(&indices, bits).unwrap());
            },
        );

        let packed = pack::pack_indices(&indices, bits).unwrap();
        group.bench_with_input(
            BenchmarkId::new("unpack", format!("{dim}x{bits}b")),
            &dim,
            |bench, _| {
                bench.iter(|| pack::unpack_indices(&packed, bits, dim).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mse_quantize,
    bench_mse_dequantize,
    bench_mse_bits,
    bench_prod,
    bench_pack
);
criterion_main!(benches);
