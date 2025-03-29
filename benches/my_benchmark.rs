use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sequencercl::rabin_karp::compare_hashes;
use sequencercl::rabin_karp::compute_base_powers;
use sequencercl::rabin_karp::compute_hashes_opencl;
use sequencercl::rabin_karp::find_matching_substrings;

static DATA1: &[u8] = include_bytes!("data1.txt");
static DATA2: &[u8] = include_bytes!("data2.txt");

static K: usize = 8;

fn criterion_find_matching_substrings(c: &mut Criterion) {
    
    let base = 256;
    let mod_value = 1_000_000_007;

    c.bench_function("find_matching_substrings", |b| {
        b.iter(|| {
            find_matching_substrings(black_box(DATA1), black_box(DATA2), K, base, mod_value)
                .unwrap()
        })
    });
}

fn criterion_compute_hashes_opencl(c: &mut Criterion) {
    
    let base = 256;
    let mod_value = 1_000_000_007;

    let base_powers = compute_base_powers(base, K, mod_value);

    c.bench_function("compute_hashes_opencl", |b| {
        b.iter(|| {
            compute_hashes_opencl(black_box(DATA1), K, black_box(&base_powers), mod_value).unwrap()
        })
    });
}

fn criterion_compare_hashes(c: &mut Criterion) {
   
    let base = 256;
    let mod_value = 1_000_000_007;

    let base_powers = compute_base_powers(base, K, mod_value);

    let hashes1 = compute_hashes_opencl(DATA1, K, &base_powers, mod_value).unwrap();
    let hashes2 = compute_hashes_opencl(DATA2, K, &base_powers, mod_value).unwrap();

    c.bench_function("compare_hashes", |b| {
        b.iter(|| {
            compare_hashes(
                black_box(&hashes1),
                black_box(&hashes2),
                black_box(DATA1),
                black_box(DATA2),
                K,
            )
        })
    });
}

criterion_group!(
    benches,
    criterion_find_matching_substrings,
    criterion_compute_hashes_opencl,
    criterion_compare_hashes,
    // Add more benchmarks here as needed
);
criterion_main!(benches);
