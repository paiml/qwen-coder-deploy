#!/usr/bin/env -S cargo +nightly -Zscript
//! PMAT-321: WGPU matmul benchmark for Radeon Pro W5700X
//! Run: cargo script scripts/wgpu-matmul-bench.rs
//! Or: rustc scripts/wgpu-matmul-bench.rs -o /tmp/wgpu-bench && /tmp/wgpu-bench

// This is a standalone benchmark — compile against trueno with gpu feature
// Build: cd ~/src/trueno && cargo build --release --features gpu
// Then run this with the trueno lib linked

fn main() {
    // Dimensions matching Qwen2.5-Coder-1.5B forward pass
    let configs = vec![
        // (M, K, N, description)
        (1, 1536, 1536, "Q proj M=1 (decode)"),
        (1, 1536, 8960, "FFN up M=1 (decode)"),
        (1, 8960, 1536, "FFN down M=1 (decode)"),
        (23, 1536, 1536, "Q proj M=23 (short prefill)"),
        (23, 1536, 8960, "FFN up M=23 (short prefill)"),
        (102, 1536, 1536, "Q proj M=102 (medium prefill)"),
        (102, 1536, 8960, "FFN up M=102 (medium prefill)"),
    ];

    println!("WGPU Matmul Benchmark — Radeon Pro W5700X");
    println!("==========================================");
    println!("{:<35} {:>8} {:>10} {:>10}", "Config", "M×K×N", "CPU (ms)", "Note");
    println!("{:-<70}", "");

    for (m, k, n, desc) in &configs {
        let a = vec![0.1f32; m * k];
        let b = vec![0.1f32; k * n];
        let mut c = vec![0.0f32; m * n];

        // CPU baseline: naive matmul
        let start = std::time::Instant::now();
        let iters = if *m <= 1 { 100 } else { 10 };
        for _ in 0..iters {
            naive_matmul(&a, &b, &mut c, *m, *k, *n);
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let flops = 2.0 * (*m as f64) * (*k as f64) * (*n as f64);
        let gflops = flops / (cpu_ms * 1e6);

        println!(
            "{:<35} {:>8} {:>9.2}ms {:>8.1} GFLOPS",
            desc,
            format!("{}×{}×{}", m, k, n),
            cpu_ms,
            gflops
        );
    }

    println!("\nNote: WGPU benchmark requires linking trueno with --features gpu.");
    println!("CPU numbers above are naive (no SIMD). Trueno SIMD is ~4-8x faster.");
    println!("WGPU on W5700X (~9 TFLOPS) should be 10-100x faster than naive CPU.");
}

fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
