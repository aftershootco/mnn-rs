# mnn-rs

Rust wrapper over [alibaba/MNN](https://github.com/alibaba/MNN) c++ library with handwritten C wrapper over mnn

NOTES:
On windows it will only compile with --release mode
There's a few issues with rustc linking to msvcrt by default and anything compiled with /MTd will not link properly
