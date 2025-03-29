
#[cfg(test)]
mod cli_tests {
    use anyhow::Result;
    use std::io::Write;
    use assert_cmd::Command;
    use predicates::prelude::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cli_with_matching_files() -> Result<()> {
        // Create temporary files with some matching content
        let mut file1 = NamedTempFile::new()?;
        let mut file2 = NamedTempFile::new()?;
        
        file1.write_all(b"hello world\nshared content\nanother line")?;
        file2.write_all(b"this has shared content\nin the middle")?;

        let path1 = file1.path().to_str().unwrap();
        let path2 = file2.path().to_str().unwrap();

        let _build_command = Command::new("cargo")
            .arg("build")
            //.arg("--release")
            .output()
            .expect("Failed to build the project");

        let mut cmd = Command::cargo_bin(env!("CARGO_PKG_NAME"))?;
        let assert = cmd
            .arg("--file1")
            .arg(path1)
            .arg("--file2")
            .arg(path2)
            .arg("-k")
            .arg("5")
            .assert();

        assert
            .success()
            .stdout(predicate::str::contains("share"))
            .stdout(predicate::str::contains("Found"));

        Ok(())
    }

    #[test]
    fn test_cli_missing_arguments() -> Result<()> {
        let mut cmd = Command::cargo_bin("sequencercl")?;
        let assert = cmd.assert();

        assert
            .failure()
            .stderr(predicate::str::contains("file1"))
            .stderr(predicate::str::contains("file2"));

        Ok(())
    }

    #[test]
    fn test_cli_nonexistent_file() -> Result<()> {
        let mut cmd = Command::cargo_bin("sequencercl")?;
        let assert = cmd
            .arg("--file1")
            .arg("nonexistent1.txt")
            .arg("--file2")
            .arg("nonexistent2.txt")
            .assert();

        assert
            .failure()
            .stderr(predicate::str::contains("Failed to read file"));

        Ok(())
    }

    #[test]
    fn test_cli_with_custom_parameters() -> Result<()> {
        let mut file1 = NamedTempFile::new()?;
        let mut file2 = NamedTempFile::new()?;
        
        file1.write_all(b"abcde")?;
        file2.write_all(b"fghij")?;

        let path1 = file1.path().to_str().unwrap();
        let path2 = file2.path().to_str().unwrap();

        let mut cmd = Command::cargo_bin("sequencercl")?;
        let assert = cmd
            .arg("--file1")
            .arg(path1)
            .arg("--file2")
            .arg(path2)
            .arg("-k")
            .arg("3")
            .arg("--base")
            .arg("128")
            .arg("--mod-value")
            .arg("1009")
            .assert();

        assert.success();

        Ok(())
    }

    #[test]
    fn test_cli_with_k_too_large() -> Result<()> {
        let mut file1 = NamedTempFile::new()?;
        let mut file2 = NamedTempFile::new()?;
        
        file1.write_all(b"short")?;
        file2.write_all(b"short")?;

        let path1 = file1.path().to_str().unwrap();
        let path2 = file2.path().to_str().unwrap();

        let mut cmd = Command::cargo_bin("sequencercl")?;
        let assert = cmd
            .arg("--file1")
            .arg(path1)
            .arg("--file2")
            .arg(path2)
            .arg("-k")
            .arg("10")
            .assert();

        assert
            .failure()
            .stderr(predicate::str::contains("shorter than k"));

        Ok(())
    }
}