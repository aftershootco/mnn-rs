use std::path::{Path, PathBuf};

use anyhow::*;
use sha2::Digest as _;

use crate::options::{TARGET_ARCH, TARGET_OS};

/// Prefixes of the release asset names (`mnn_<version>_<suffix>.zip`) per platform. The
/// rest of the suffix lists the bundled backends and changes between releases (3.5.0 shipped
/// `linux_x64_cpu_opencl`, 3.6.x ships `linux_x64_cpu_opencl_vulkan`), so the full name is
/// looked up in `checksums.json` instead of being hard coded.
pub const PLATFORM_PREFIXES: [&str; 5] = ["android_", "ios_", "linux_x64", "windows_x64", "macos_"];

const CHECKSUMS_JSON: &str = include_str!("checksums.json");

/// Marker written into an extraction directory once the archive has been fully extracted,
/// so that a partially extracted directory (interrupted build) is never reused.
const EXTRACTED_MARKER: &str = ".mnn-rs-extracted";

/// Look up the release asset suffix and its SHA-256 digest recorded in `checksums.json` for
/// the platform whose asset name starts with `platform_prefix`.
///
/// When the json's version field doesn't match the requested MNN version the asset name
/// recorded for the known version is reused (the names rarely change between releases) and
/// the digest is `"sha256:placeholder"`, signalling `verify_checksum` to skip the check.
fn asset_for(version: &str, platform_prefix: &str) -> Result<(String, String)> {
    let value: serde_json::Value = serde_json::from_str(CHECKSUMS_JSON)
        .context("Failed to parse mnn-sys/build/checksums.json")?;
    let recorded_version = value
        .get("version")
        .and_then(|v| v.as_str())
        .context("checksums.json missing 'version' field")?;
    let (suffix, digest) = value
        .get("checksums")
        .and_then(|c| c.as_object())
        .context("checksums.json missing 'checksums' object")?
        .iter()
        .find(|(name, _)| name.starts_with(platform_prefix))
        .with_context(|| {
            format!(
                "checksums.json has no asset starting with '{}' (run `just checksums {}`)",
                platform_prefix, version
            )
        })?;
    let digest = digest
        .as_str()
        .with_context(|| format!("checksums.json digest for '{}' is not a string", suffix))?;
    if recorded_version != version {
        eprintln!(
            "warning: checksums.json is for MNN {} but MNN_VERSION={}; reusing asset name '{}' and skipping checksum verification",
            recorded_version, version, suffix
        );
        return Ok((suffix.clone(), "sha256:placeholder".to_string()));
    }
    Ok((suffix.clone(), digest.to_string()))
}

const USER_AGENT: &str = concat!("mnn-rs-build/", env!("CARGO_PKG_VERSION"));

fn create_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        // The blocking client defaults to a 30s total timeout, which the multi hundred
        // megabyte release archives easily exceed on slower connections.
        .timeout(None)
        .connect_timeout(std::time::Duration::from_secs(60))
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

/// Download `url` to `dest_file` unless a file with the expected checksum is already there.
///
/// A cached file that fails verification (for example a leftover from an interrupted
/// download) is removed and downloaded again instead of failing every subsequent build.
pub fn download_file(url: &str, dest_file: &Path, checksum: &str) -> Result<()> {
    if dest_file.exists() {
        eprintln!(
            "File already exists at {}, verifying checksum",
            dest_file.display()
        );
        match verify_checksum(dest_file, checksum) {
            Result::Ok(()) => {
                eprintln!("File verified, skipping download");
                return Ok(());
            }
            Err(err) => {
                eprintln!(
                    "warning: cached file {} failed verification ({err:#}); downloading again",
                    dest_file.display()
                );
                std::fs::remove_file(dest_file).with_context(|| {
                    format!("Failed to remove stale file {}", dest_file.display())
                })?;
            }
        }
    }

    let client = create_client()?;
    let mut response = client
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

    // Stream straight to disk; the archives are too large to buffer comfortably.
    let partial = dest_file.with_extension("part");
    let mut file = std::fs::File::create(&partial)
        .with_context(|| format!("Failed to create {}", partial.display()))?;
    std::io::copy(&mut response, &mut file)
        .with_context(|| format!("Failed to download {} to {}", url, partial.display()))?;
    drop(file);
    std::fs::rename(&partial, dest_file).with_context(|| {
        format!(
            "Failed to move {} to {}",
            partial.display(),
            dest_file.display()
        )
    })?;

    if let Err(err) = verify_checksum(dest_file, checksum) {
        // Don't leave a file behind that would be "verified" again on the next build.
        let _ = std::fs::remove_file(dest_file);
        return Err(err).with_context(|| {
            format!(
                "Checksum verification failed for downloaded file at {}, expected checksum: {}",
                dest_file.display(),
                checksum
            )
        });
    }

    Ok(())
}

/// Extract `zip_path` into `dest`, stripping the single top level directory every MNN
/// archive is wrapped in (its name differs between platforms and versions).
///
/// Extraction is skipped when `dest` already holds a complete extraction of the archive.
fn extract_zip(zip_path: &Path, dest: &Path) -> Result<()> {
    let marker = dest.join(EXTRACTED_MARKER);
    if marker.exists() {
        return Ok(());
    }
    if dest.exists() {
        std::fs::remove_dir_all(dest).with_context(|| {
            format!(
                "Failed to remove partially extracted directory {}",
                dest.display()
            )
        })?;
    }

    let file = std::fs::File::open(zip_path).with_context(|| {
        format!(
            "Failed to open zip file at {} for extraction",
            zip_path.display()
        )
    })?;

    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip archive from {}", zip_path.display()))?;

    zip.extract_unwrapped_root_dir(dest, |_: &Path| true)
        .with_context(|| format!("Failed to extract archive to {}", dest.display()))?;

    std::fs::write(&marker, zip_path.display().to_string())
        .with_context(|| format!("Failed to write {}", marker.display()))?;

    Ok(())
}

/// The download url and expected checksum of the prebuilt archive for the target.
pub fn url_and_checksum(version: impl AsRef<str>) -> Result<(String, String)> {
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

    let (suffix, checksum) = asset_for(version, PLATFORM_PREFIXES[idx])?;
    Ok((format!("{}_{}.zip", pre_url, suffix), checksum))
}

/// Download and extract the prebuilt MNN release for the target, returning the directory
/// that holds the extracted archive (with the archive's top level directory stripped).
///
/// All the paths are versioned so that changing `MNN_VERSION` never reuses a stale download.
pub fn download_prebuilt_mnn(
    version: impl AsRef<str>,
    out_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let version = version.as_ref();
    let (url, checksum) = url_and_checksum(version)?;
    let dest = out_dir.as_ref().join(format!("mnn_{version}_prebuilt"));
    let dest_file = out_dir.as_ref().join(format!("mnn_{version}_prebuilt.zip"));

    download_file(&url, &dest_file, &checksum)?;
    extract_zip(&dest_file, &dest)?;

    Ok(dest)
}

/// Download and extract the MNN source tree for `version` (needed for the headers that the
/// C bridge and bindgen consume), returning the directory that holds it.
pub fn download_mnn_source(version: impl AsRef<str>, out_dir: impl AsRef<Path>) -> Result<PathBuf> {
    let version = version.as_ref();
    let url = format!(
        "https://api.github.com/repos/alibaba/MNN/zipball/{}",
        version
    );
    let dest = out_dir.as_ref().join(format!("mnn_{version}_source"));
    let dest_file = out_dir.as_ref().join(format!("mnn_{version}_source.zip"));

    // GitHub generates the source archive on the fly, so there is no stable digest to check.
    download_file(&url, &dest_file, "sha256:placeholder")?;
    extract_zip(&dest_file, &dest)?;

    if !dest.join("include").join("MNN").is_dir() {
        anyhow::bail!(
            "Extracted MNN source at {} does not contain include/MNN",
            dest.display()
        );
    }

    Ok(dest)
}

/// Emit the link directives for the extracted prebuilt archive at `prebuilt_dir`.
///
/// Layouts (verified against the 3.5.0 and 3.6.1 release assets):
/// - android: `<abi>/libMNN.so` (+ `libMNN_CL.so`, `libMNN_Vulkan.so`, `libmnncore.so`, `libc++_shared.so`)
/// - ios:     `Static/MNN.framework` (static framework)
/// - linux:   `lib/{Debug,Release}/libMNN.{a,so}`
/// - windows: `lib/x64/Release/{Static,Dynamic}/{MT,MD}/MNN.lib`
/// - macos:   `{Static,Dynamic}/MNN.framework` (fat x86_64 + arm64)
pub fn prebuilt_lib_link(prebuilt_dir: &Path) -> Result<()> {
    use build_target::Arch;

    // `cfg!(debug_assertions)` describes the build script, not the crate being built.
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let debug_string = if profile == "debug" {
        "Debug"
    } else {
        "Release"
    };

    match (&*TARGET_ARCH, &*TARGET_OS) {
        (Arch::AArch64 | Arch::Arm, build_target::Os::Android) => {
            let arch = if *TARGET_ARCH == Arch::Arm {
                "armeabi-v7a"
            } else {
                "arm64-v8a"
            };
            println!(
                "cargo:rustc-link-search=native={}",
                prebuilt_dir.join(arch).display()
            );
            println!("cargo:rustc-link-lib=dylib=MNN");
            println!("cargo:rustc-link-lib=dylib=MNN_Vulkan");
            println!("cargo:rustc-link-lib=dylib=MNN_CL");
            println!("cargo:rustc-link-lib=dylib=c++_shared");
            println!("cargo:rustc-link-lib=dylib=mnncore");
        }
        (Arch::AArch64, build_target::Os::iOS) => {
            println!(
                "cargo:rustc-link-search=framework={}",
                prebuilt_dir.join("Static").display()
            );
            println!("cargo:rustc-link-lib=framework=MNN");
            // The prebuilt framework contains the Metal and CoreML backends.
            for framework in ["Foundation", "CoreGraphics", "Metal", "CoreML", "CoreVideo"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        (Arch::X86_64, build_target::Os::Linux) => {
            println!(
                "cargo:rustc-link-search=native={}",
                prebuilt_dir.join("lib").join(debug_string).display()
            );
            println!("cargo:rustc-link-lib=static=MNN");
        }
        (Arch::X86_64, build_target::Os::Windows) => {
            let crt = if cfg!(feature = "crt_static") {
                "MT"
            } else {
                "MD"
            };
            println!(
                "cargo:rustc-link-search=native={}",
                prebuilt_dir
                    .join("lib")
                    .join("x64")
                    .join("Release")
                    .join("Static")
                    .join(crt)
                    .display()
            );
            println!("cargo:rustc-link-lib=static=MNN");
        }
        (Arch::X86_64 | Arch::AArch64, build_target::Os::MacOS) => {
            println!(
                "cargo:rustc-link-search=framework={}",
                prebuilt_dir.join("Static").display()
            );
            println!("cargo:rustc-link-lib=framework=MNN");
            // The prebuilt framework contains the Metal and OpenCL backends.
            for framework in ["Foundation", "CoreGraphics", "Metal", "OpenCL"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        (arch, os) => anyhow::bail!("Prebuilt MNN is not available for target {}-{}", arch, os),
    };
    Ok(())
}
