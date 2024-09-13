# mnn-rs
![Codecov](https://img.shields.io/codecov/c/github/aftershootco/mnn-rs?link=https%3A%2F%2Fapp.codecov.io%2Fgithub%2Faftershootco%2Fmnn-rs)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aftershootco/mnn-rs/build.yaml?link=https%3A%2F%2Fgithub.com%2Faftershootco%2Fmnn-rs%2Factions%2Fworkflows%2Fbuild.yaml)

Rust wrapper over [alibaba/MNN](https://github.com/alibaba/MNN) c++ library with handwritten C wrapper over mnn

If you have nix you can just build the inspect binary with

```
nix build .#inspect
```

NOTES:
On windows it will only compile with --release mode
There's a few issues with rustc linking to msvcrt by default and anything compiled with /MTd will not link properly



