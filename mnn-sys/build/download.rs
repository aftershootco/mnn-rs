use std::path::Path;

use anyhow::*;
use sha2::Digest as _;

use crate::options::{CHECKSUMS, SUFFIXES, TARGET_ARCH, TARGET_OS};

const USER_AGENT: &str = concat!("mnn-rs-build/", env!("CARGO_PKG_VERSION"));

fn create_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .context("Failed to create HTTP client")
}

pub fn verify_checksum(path: impl AsRef<Path>, expected: impl AsRef<str>) -> Result<()> {
    let expected = expected.as_ref();
    if expected == "sha256:placeholder" {
        return Ok(());
    }
    let mut file = std::fs::File::open(&path).with_context(|| {
        format!(
            "Failed to open file for checksum verification: {}",
            path.as_ref().display()
        )
    })?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).with_context(|| {
        format!(
            "Failed to read file for checksum verification: {}",
            path.as_ref().display()
        )
    })?;
    let actual = format!("sha256:{:x}", hasher.finalize());
    if actual != expected {
        anyhow::bail!(
            "Checksum mismatch for {}: expected {}, got {}",
            path.as_ref().display(),
            expected,
            actual
        );
    }
    Ok(())
}

pub fn download_file(url: &str, dest_file: &Path, checksum: &str) -> Result<()> {
    if dest_file.exists() {
        eprintln!(
            "File already exists at {}, verifying checksum",
            dest_file.display()
        );
        verify_checksum(dest_file, checksum).with_context(|| {
            format!(
                "Checksum verification failed for existing file at {}, expected checksum: {}",
                dest_file.display(),
                checksum
            )
        })?;
        eprintln!("File verified, skipping download");
        return Ok(());
    }

    let client = create_client()?;
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to download from {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download from {}, status: {}",
            url,
            response.status()
        );
    }

    let bytes = response
        .bytes()
        .with_context(|| format!("Failed to read response bytes from {}", url))?;

    std::fs::write(dest_file, &bytes).with_context(|| {
        format!(
            "Failed to save file from {} to {}",
            url,
            dest_file.display()
        )
    })?;

    verify_checksum(dest_file, checksum).with_context(|| {
        format!(
            "Checksum verification failed for downloaded file at {}, expected checksum: {}",
            dest_file.display(),
            checksum
        )
    })?;

    Ok(())
}

fn extract_zip(zip_path: &Path, dest: &Path, root_filter: Option<&str>) -> Result<()> {
    let file = std::fs::File::open(zip_path).with_context(|| {
        format!(
            "Failed to open zip file at {} for extraction",
            zip_path.display()
        )
    })?;

    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip archive from {}", zip_path.display()))?;

    if let Some(root) = root_filter {
        zip.extract_unwrapped_root_dir(dest, |path| path == Path::new(root))
            .with_context(|| format!("Failed to extract archive to {}", dest.display()))?;
    } else {
        zip.extract(dest)
            .with_context(|| format!("Failed to extract archive to {}", dest.display()))?;
    }

    Ok(())
}

pub fn url_name_checksum(version: impl AsRef<str>) -> Result<(String, String, String)> {
    let version = version.as_ref();
    let pre_url =
        format!("https://github.com/alibaba/MNN/releases/download/{version}/mnn_{version}");

    let idx = match (&*TARGET_ARCH, &*TARGET_OS) {
        (build_target::Arch::AArch64 | build_target::Arch::Arm, build_target::Os::Android) => 0,
        (build_target::Arch::AArch64, build_target::Os::iOS) => 1,
        (build_target::Arch::X86_64, build_target::Os::Linux) => 2,
        (build_target::Arch::X86_64, build_target::Os::Windows) => 3,
        (build_target::Arch::X86_64 | build_target::Arch::AArch64, build_target::Os::MacOS) => 4,
        (arch, os) => anyhow::bail!("Prebuilt MNN is not available for target {}-{}", arch, os),
    };

    Ok((
        format!("{}_{}.zip", pre_url, SUFFIXES[idx]),
        format!("mnn_{version}_{}", SUFFIXES[idx]),
        CHECKSUMS[idx].to_string(),
    ))
}

pub fn download_prebuilt_mnn(version: impl AsRef<str>, out_dir: impl AsRef<Path>) -> Result<()> {
    let (url, root, checksum) = url_name_checksum(version)?;
    let dest = out_dir.as_ref().join("mnn_prebuilt");
    let dest_file = out_dir.as_ref().join("mnn_prebuilt.zip");

    download_file(&url, &dest_file, &checksum)?;
    extract_zip(&dest_file, &dest, Some(&root))?;

    Ok(())
}

pub fn download_mnn_source(
    version: impl AsRef<str>,
    out_dir: impl AsRef<Path>,
) -> Result<std::path::PathBuf> {
    let version = version.as_ref();
    let url = format!(
        "https://api.github.com/repos/alibaba/MNN/zipball/{}",
        version
    );
    let dest = out_dir.as_ref().join("mnn_source");
    let dest_file = out_dir.as_ref().join("mnn_source.zip");

    download_file(&url, &dest_file, "sha256:placeholder")?;

    if dest.exists() {
        if let Some(subdir) = dest.read_dir()?.flatten().find(|e| e.path().is_dir()) {
            return Ok(subdir.path());
        }
    }

    extract_zip(&dest_file, &dest, None)?;

    let subdir = dest
        .read_dir()?
        .flatten()
        .find(|e| e.path().is_dir())
        .map(|e| e.path())
        .with_context(|| {
            format!(
                "Failed to find extracted source directory in {}",
                dest.display()
            )
        })?;

    Ok(subdir)
}
