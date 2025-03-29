use anyhow::Result;
use ocl::{Buffer, Kernel, Program, Queue};
use std::collections::HashMap;

/// Computes the base powers for Rabin-Karp algorithm.
/// The base powers are used to calculate the hash values of substrings.
pub fn compute_base_powers(base: u64, k: usize, mod_value: u64) -> Vec<u64> {
    let mut powers = Vec::with_capacity(k);
    let mut current = 1u64;
    for _ in 0..k {
        powers.push(current);
        current = (current * base) % mod_value;
    }
    powers.reverse();
    powers
}

/// Computes the Rabin-Karp hashes of all substrings of length `k` in the given data using OpenCL.
/// The hashes are computed using the base powers and modulo value provided.
/// The function uses OpenCL to parallelize the hash computation for better performance.
/// If `k` is 0, returns an error. If the data length is shorter than `k`, returns an error.
/// Returns a vector of hashes.
pub fn compute_hashes_opencl(
    data: &[u8],
    k: usize,
    base_powers: &[u64],
    mod_value: u64,
) -> Result<Vec<u64>> {
    if k == 0 {
        return Err(anyhow::anyhow!("k must be at least 1"));
    }

    if data.len() < k - 1 {
        return Err(anyhow::anyhow!("data lengnth is shorter than k"));
    }

    let data_len = data.len();
    let hashes_len = data_len - k + 1;

    if hashes_len == 0 {
        return Ok(Vec::new());
    }

    let src = r#"
        __kernel void compute_hashes(
            __global const uchar* data,
            uint data_length,
            uint k,
            __global const ulong* base_powers,
            ulong mod_value,
            __global ulong* hashes
        ) {
            uint i = get_global_id(0);
            if (i + k > data_length) {
                return;
            }
            
            ulong hash = 0;
            for (uint j = 0; j < k; j++) {
                hash += (ulong)data[i + j] * base_powers[j];
            }
            hash %= mod_value;
            hashes[i] = hash;
        }
    "#;

    let platform = ocl::Platform::default();
    let device = ocl::Device::first(platform)?;
    let context = ocl::Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&context, device, None)?;
    let program = Program::builder()
        .src(src)
        .devices(device)
        .build(&context)?;

    let data_buffer = Buffer::builder()
        .queue(queue.clone())
        .len(data.len())
        .copy_host_slice(data)
        .build()?;

    let base_powers_buffer = Buffer::builder()
        .queue(queue.clone())
        .len(base_powers.len())
        .copy_host_slice(base_powers)
        .build()?;

    let hashes_buffer = Buffer::builder()
        .queue(queue.clone())
        .len(hashes_len)
        .build()?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("compute_hashes")
        .queue(queue.clone())
        .global_work_size(hashes_len)
        .arg(&data_buffer)
        .arg(&(data_len as u32))
        .arg(&(k as u32))
        .arg(&base_powers_buffer)
        .arg(mod_value)
        .arg(&hashes_buffer)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut hashes = vec![0; hashes_len];
    hashes_buffer.read(&mut hashes).enq()?;

    Ok(hashes)
}

pub fn compare_hashes(
    hashes1: &[u64],
    hashes2: &[u64],
    data1: &[u8],
    data2: &[u8],
    k: usize,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();

    let mut map1: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &hash) in hashes1.iter().enumerate() {
        map1.entry(hash).or_default().push(i);
    }

    let mut map2: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &hash) in hashes2.iter().enumerate() {
        map2.entry(hash).or_default().push(i);
    }

    for (hash, indices1) in map1 {
        if let Some(indices2) = map2.get(&hash) {
            for &i in &indices1 {
                let substring1 = &data1[i..i + k];
                for &j in indices2 {
                    let substring2 = &data2[j..j + k];
                    if substring1 == substring2 {
                        matches.push((i, j));
                    }
                }
            }
        }
    }

    matches
}

/// Finds matching substrings of length `k` in two byte arrays using Rabin-Karp algorithm with OpenCL.  
/// The function computes the hashes of all substrings of length `k` in both arrays and finds matches.
/// Returns a vector of tuples containing the starting indices of matching substrings in both arrays.
pub fn find_matching_substrings(
    data1: &[u8],
    data2: &[u8],
    k: usize,
    base: u64,
    mod_value: u64,
) -> Result<Vec<(usize, usize)>> {
    if k == 0 {
        return Err(anyhow::anyhow!("k must be at least 1"));
    }
    if data1.len() < k || data2.len() < k {
        return Err(anyhow::anyhow!("File is shorter than k"));
    }

    let base_powers = compute_base_powers(base, k, mod_value);

    let hashes1 = compute_hashes_opencl(data1, k, &base_powers, mod_value)?;
    let hashes2 = compute_hashes_opencl(data2, k, &base_powers, mod_value)?;

    let matches = compare_hashes(&hashes1, &hashes2, data1, data2, k);

    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_base_powers() {
        let powers = compute_base_powers(256, 3, 1_000_000_007);
        assert_eq!(powers.len(), 3);
        assert_eq!(powers[0], 65536); // 256^2
        assert_eq!(powers[1], 256); // 256^1
        assert_eq!(powers[2], 1); // 256^0
    }

    #[test]
    fn test_compute_hashes_opencl_small_input() {
        let data = b"abcde";
        let k = 3;
        let base = 256;
        let mod_value = 1_000_000_007;
        let base_powers = compute_base_powers(base, k, mod_value);

        let hashes = compute_hashes_opencl(data, k, &base_powers, mod_value).unwrap();
        assert_eq!(hashes.len(), 3); // 5-3+1=3 substrings

        // Manual calculation of expected hashes
        let expected1 = ((b'a' as u64) * 65536 + (b'b' as u64) * 256 + (b'c' as u64)) % mod_value;
        let expected2 = ((b'b' as u64) * 65536 + (b'c' as u64) * 256 + (b'd' as u64)) % mod_value;
        let expected3 = ((b'c' as u64) * 65536 + (b'd' as u64) * 256 + (b'e' as u64)) % mod_value;

        assert_eq!(hashes[0], expected1);
        assert_eq!(hashes[1], expected2);
        assert_eq!(hashes[2], expected3);
    }

    #[test]
    fn test_find_matching_substrings_basic() {
        let data1 = b"hello world";
        let data2 = b"world hello";
        let k = 5;
        let base = 256;
        let mod_value = 1_000_000_007;

        let matches = find_matching_substrings(data1, data2, k, base, mod_value).unwrap();

        // "hello" appears in both at positions 0 and 6 respectively
        assert!(matches.contains(&(0, 6)));
        // "world" appears in both at positions 6 and 0 respectively
        assert!(matches.contains(&(6, 0)));
    }

    #[test]
    fn test_find_matching_substrings_no_matches() {
        let data1 = b"abcdef";
        let data2 = b"ghijkl";
        let k = 3;
        let base = 256;
        let mod_value = 1_000_000_007;

        let matches = find_matching_substrings(data1, data2, k, base, mod_value).unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_find_matching_substrings_k_too_large() {
        let data1 = b"abc";
        let data2 = b"def";
        let k = 5;
        let base = 256;
        let mod_value = 1_000_000_007;

        let result = find_matching_substrings(data1, data2, k, base, mod_value);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_matching_substrings_empty_input() {
        let data1 = b"";
        let data2 = b"test";
        let k = 1;
        let base = 256;
        let mod_value = 1_000_000_007;

        let result = find_matching_substrings(data1, data2, k, base, mod_value);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_matching_substrings_multiple_matches() {
        let data1 = b"ababab";
        let data2 = b"bababa";
        let k = 2;
        let base = 256;
        let mod_value = 1_000_000_007;

        let matches = find_matching_substrings(data1, data2, k, base, mod_value).unwrap();
        assert_eq!(matches.len(), 12); // Multiple overlapping matches

        println!("{:?}", matches);
    }
}
