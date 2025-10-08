// Async tests with tokio to diagnose interpreter creation issues

#[cfg(test)]
#[tokio::test]
async fn test_from_file_in_tokio_direct() {
    println!("\n=== Test: from_file called directly in tokio async context ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");

    println!("Tokio thread: {:?}", std::thread::current().id());

    match mnn::Interpreter::from_file(model_path) {
        Ok(_interpreter) => {
            println!("✓ Direct async call succeeded");
        }
        Err(e) => {
            println!("✗ Direct async call failed: {:?}", e);
            panic!("Direct async call failed");
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_from_bytes_in_tokio_direct() {
    println!("\n=== Test: from_bytes called directly in tokio async context ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");
    let bytes = std::fs::read(model_path).expect("Failed to read model file");

    println!("Tokio thread: {:?}", std::thread::current().id());
    println!("Model file size: {} bytes", bytes.len());

    match mnn::Interpreter::from_bytes(&bytes) {
        Ok(_interpreter) => {
            println!("✓ Direct async call succeeded");
        }
        Err(e) => {
            println!("✗ Direct async call failed: {:?}", e);
            panic!("Direct async call failed");
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_from_file_with_spawn_blocking() {
    println!("\n=== Test: from_file with tokio::task::spawn_blocking ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn").to_path_buf();

    println!("Main tokio thread: {:?}", std::thread::current().id());

    match tokio::task::spawn_blocking(move || {
        println!("Blocking thread: {:?}", std::thread::current().id());
        mnn::Interpreter::from_file(model_path)
    })
    .await
    {
        Ok(Ok(_interpreter)) => {
            println!("✓ spawn_blocking succeeded");
        }
        Ok(Err(e)) => {
            println!("✗ spawn_blocking failed: {:?}", e);
            panic!("spawn_blocking failed");
        }
        Err(e) => {
            println!("✗ Task join failed: {:?}", e);
            panic!("Task join failed");
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_from_bytes_with_spawn_blocking() {
    println!("\n=== Test: from_bytes with tokio::task::spawn_blocking ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");
    let bytes = std::fs::read(model_path).expect("Failed to read model file");

    println!("Main tokio thread: {:?}", std::thread::current().id());
    println!("Model file size: {} bytes", bytes.len());

    match tokio::task::spawn_blocking(move || {
        println!("Blocking thread: {:?}", std::thread::current().id());
        mnn::Interpreter::from_bytes(bytes)
    })
    .await
    {
        Ok(Ok(_interpreter)) => {
            println!("✓ spawn_blocking succeeded");
        }
        Ok(Err(e)) => {
            println!("✗ spawn_blocking failed: {:?}", e);
            panic!("spawn_blocking failed");
        }
        Err(e) => {
            println!("✗ Task join failed: {:?}", e);
            panic!("Task join failed");
        }
    }
}

#[cfg(test)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_from_file_multi_thread_runtime() {
    println!("\n=== Test: from_file in multi-threaded tokio runtime ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");

    println!("Main tokio thread: {:?}", std::thread::current().id());

    match mnn::Interpreter::from_file(model_path) {
        Ok(_interpreter) => {
            println!("✓ Multi-thread runtime succeeded");
        }
        Err(e) => {
            println!("✗ Multi-thread runtime failed: {:?}", e);
            panic!("Multi-thread runtime failed");
        }
    }
}

#[cfg(test)]
#[tokio::test(flavor = "current_thread")]
async fn test_from_file_current_thread_runtime() {
    println!("\n=== Test: from_file in current-thread tokio runtime ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");

    println!("Main tokio thread: {:?}", std::thread::current().id());

    match mnn::Interpreter::from_file(model_path) {
        Ok(_interpreter) => {
            println!("✓ Current-thread runtime succeeded");
        }
        Err(e) => {
            println!("✗ Current-thread runtime failed: {:?}", e);
            panic!("Current-thread runtime failed");
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_multiple_concurrent_from_file() {
    println!("\n=== Test: Multiple concurrent from_file calls ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");

    let mut handles = vec![];

    for i in 0..3 {
        let path = model_path.to_path_buf();
        let handle = tokio::task::spawn_blocking(move || {
            println!("Task {} on thread: {:?}", i, std::thread::current().id());
            let result = mnn::Interpreter::from_file(path);
            println!("Task {} result: {}", i, result.is_ok());
            result
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(Ok(_)) => {
                println!("✓ Task {} succeeded", i);
                success_count += 1;
            }
            Ok(Err(e)) => {
                println!("✗ Task {} failed: {:?}", i, e);
            }
            Err(e) => {
                println!("✗ Task {} join failed: {:?}", i, e);
            }
        }
    }

    println!("\n{}/3 tasks succeeded", success_count);

    if success_count < 3 {
        panic!("Some tasks failed");
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_from_file_after_await_point() {
    println!("\n=== Test: from_file after an .await point ===");
    let model_path = std::path::Path::new("tests/assets/realesr.mnn");

    println!("Before await - thread: {:?}", std::thread::current().id());

    // 引入一个 await 点
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    println!("After await - thread: {:?}", std::thread::current().id());

    match mnn::Interpreter::from_file(model_path) {
        Ok(_interpreter) => {
            println!("✓ After await point succeeded");
        }
        Err(e) => {
            println!("✗ After await point failed: {:?}", e);
            panic!("After await point failed");
        }
    }
}
