// Tests to diagnose async interpreter creation issues

// ============================================================================
// Synchronous baseline tests
// ============================================================================

#[cfg(test)]
#[test]
fn test_from_file_sync_baseline() {
    println!("\n=== Baseline test: from_file in synchronous context ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");

    match mnn::Interpreter::from_file(model_path) {
        Ok(_interpreter) => {
            println!("✓ Sync call succeeded");
        }
        Err(e) => {
            println!("✗ Sync call failed: {:?}", e);
            panic!("Baseline test failed");
        }
    }
}

#[cfg(test)]
#[test]
fn test_from_bytes_sync_baseline() {
    println!("\n=== Baseline test: from_bytes in synchronous context ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");
    let bytes = std::fs::read(model_path).expect("Failed to read model file");

    println!("Model file size: {} bytes", bytes.len());

    match mnn::Interpreter::from_bytes(&bytes) {
        Ok(_interpreter) => {
            println!("✓ Sync call succeeded");
        }
        Err(e) => {
            println!("✗ Sync call failed: {:?}", e);
            panic!("Baseline test failed");
        }
    }
}

#[cfg(test)]
#[test]
fn test_from_file_in_thread() {
    println!("\n=== Test: from_file in std::thread::spawn ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn").to_path_buf();

    let handle = std::thread::spawn(move || {
        println!("Thread ID: {:?}", std::thread::current().id());
        mnn::Interpreter::from_file(model_path)
    });

    match handle.join() {
        Ok(Ok(_interpreter)) => {
            println!("✓ Thread spawn succeeded");
        }
        Ok(Err(e)) => {
            println!("✗ Thread spawn failed: {:?}", e);
            panic!("Thread test failed");
        }
        Err(e) => {
            println!("✗ Thread panic: {:?}", e);
            panic!("Thread panicked");
        }
    }
}

#[cfg(test)]
#[test]
fn test_from_bytes_in_thread() {
    println!("\n=== Test: from_bytes in std::thread::spawn ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");
    let bytes = std::fs::read(model_path).expect("Failed to read model file");

    println!("Model file size: {} bytes", bytes.len());

    let handle = std::thread::spawn(move || {
        println!("Thread ID: {:?}", std::thread::current().id());
        mnn::Interpreter::from_bytes(bytes)
    });

    match handle.join() {
        Ok(Ok(_interpreter)) => {
            println!("✓ Thread spawn succeeded");
        }
        Ok(Err(e)) => {
            println!("✗ Thread spawn failed: {:?}", e);
            panic!("Thread test failed");
        }
        Err(e) => {
            println!("✗ Thread panic: {:?}", e);
            panic!("Thread panicked");
        }
    }
}

#[cfg(test)]
#[test]
fn test_multiple_threads_from_file() {
    println!("\n=== Test: Multiple threads creating interpreters from file ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let path = model_path.to_path_buf();
            std::thread::spawn(move || {
                println!("Thread {} starting", i);
                let result = mnn::Interpreter::from_file(path);
                println!("Thread {} result: {}", i, result.is_ok());
                result
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .enumerate()
        .map(|(i, h)| match h.join() {
            Ok(Ok(_)) => {
                println!("✓ Thread {} succeeded", i);
                true
            }
            Ok(Err(e)) => {
                println!("✗ Thread {} failed: {:?}", i, e);
                false
            }
            Err(e) => {
                println!("✗ Thread {} panicked: {:?}", i, e);
                false
            }
        })
        .collect();

    let success_count = results.iter().filter(|&&x| x).count();
    println!("\n{}/{} threads succeeded", success_count, results.len());

    if success_count < results.len() {
        panic!("Some threads failed to create interpreter");
    }
}

#[cfg(test)]
#[test]
fn test_multiple_threads_from_bytes() {
    println!("\n=== Test: Multiple threads creating interpreters from bytes ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");
    let bytes = std::fs::read(model_path).expect("Failed to read model file");

    println!("Model file size: {} bytes", bytes.len());

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let bytes_clone = bytes.clone();
            std::thread::spawn(move || {
                println!("Thread {} starting", i);
                let result = mnn::Interpreter::from_bytes(bytes_clone);
                println!("Thread {} result: {}", i, result.is_ok());
                result
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .enumerate()
        .map(|(i, h)| match h.join() {
            Ok(Ok(_)) => {
                println!("✓ Thread {} succeeded", i);
                true
            }
            Ok(Err(e)) => {
                println!("✗ Thread {} failed: {:?}", i, e);
                false
            }
            Err(e) => {
                println!("✗ Thread {} panicked: {:?}", i, e);
                false
            }
        })
        .collect();

    let success_count = results.iter().filter(|&&x| x).count();
    println!("\n{}/{} threads succeeded", success_count, results.len());

    if success_count < results.len() {
        panic!("Some threads failed to create interpreter");
    }
}
