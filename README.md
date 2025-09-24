# slurm_benchmark

A minimal Python helper for benchmarking applications on Slurm clusters.

## Usage

```python
from slurm_benchmark import run_benchmarks

command = ["python", "my_program.py"]
parameter_grid = {
    "memory": ["2G", "4G", "8G"],
    "cpus": [1, 2, 4],
}
inputs = [
    {"name": "small", "value": "data/small.txt"},
    {"name": "medium", "value": "data/medium.txt"},
    {"name": "large", "value": "data/large.txt"},
]

results = run_benchmarks(command, parameter_grid, inputs)
for result in results:
    print(result.job_id, result.state, result.time_metrics.get("Elapsed (wall clock) time"))
```

Each benchmark run receives its own directory containing the generated
Slurm script, raw command output, and logs produced by `/usr/bin/time`.
The helper also queries `sacct` for accounting information, which is
made available via `BenchmarkResult.sacct_records`.
