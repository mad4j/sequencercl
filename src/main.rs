use sequencercl::rabin_karp::find_matching_substrings;

use std::fs;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    file1: String,

    #[arg(long)]
    file2: String,

    #[arg(short, default_value_t = 5)]
    k: usize,

    #[arg(long, default_value_t = 256)]
    base: u64,

    #[arg(long, default_value_t = 1_000_000_007)]
    mod_value: u64,
}
fn main() -> Result<()> {
    let args = Args::parse();

    let data1 = fs::read(&args.file1).context("Failed to read file1")?;
    let data2 = fs::read(&args.file2).context("Failed to read file2")?;

    let matches = find_matching_substrings(&data1, &data2, args.k, args.base, args.mod_value)?;

    println!("Found {} matching substrings:", matches.len());
    for (i, j) in matches {
        let substring = String::from_utf8_lossy(&data1[i..i + args.k]);
        println!(
            "File1[{}..{}] == File2[{}..{}]: {:?}",
            i,
            i + args.k,
            j,
            j + args.k,
            substring
        );
    }

    Ok(())
}
