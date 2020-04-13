# Benchmarks

This folder contains the necessary code to define and execute benchmarks, for
example to compare the performance of two different versions (branches, commits,
tags) of the package on a predefined tasks.

Benchmarks are defined in [benchmarks.jl](./benchmarks.jl), using the tooling
provided by [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl).
You can executed benchmarks by running the [runbenchmarks.jl](./runbenchmarks.jl)
script, which makes heavy use of the tooling offered by
[PkgBenchmark](https://github.com/JuliaCI/PkgBenchmark.jl) instead.

To simply run benchmarks, execute the following command from the package root directory:

```sh
julia --project=benchmark benchmark/runbenchmarks.jl
```

To know more about the available options, use the `--help` option:

```sh
julia --project=benchmark benchmark/runbenchmarks.jl --help
```
