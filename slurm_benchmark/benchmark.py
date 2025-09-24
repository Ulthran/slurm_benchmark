"""Slurm benchmarking helpers.

This module provides a small abstraction for sweeping over resource
requirements for a given command on a Slurm cluster.  The implementation
focuses on transparency and does not hide the fact that it ultimately
constructs and submits ``sbatch`` jobs that run ``/usr/bin/time`` around
user supplied commands.  The expectation is that users inspect the
``run_directory`` referenced by :class:`BenchmarkResult` for the raw
artifacts produced by Slurm.
"""

from __future__ import annotations

import itertools
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_ACTIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}
_DEFAULT_SACCT_FORMAT = [
    "JobID",
    "JobName",
    "State",
    "Elapsed",
    "CPUTime",
    "MaxRSS",
    "MaxVMSize",
    "ReqMem",
    "ReqCPUS",
]
_DEFAULT_RESOURCE_TEMPLATES = {
    "memory": "--mem={value}",
    "mem": "--mem={value}",
    "cpus": "--cpus-per-task={value}",
    "cpus_per_task": "--cpus-per-task={value}",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "run"


@dataclass(frozen=True)
class BenchmarkInput:
    """Input definition for a benchmark run.

    Parameters
    ----------
    name:
        Human readable label for the input.  The value is also used when
        constructing directories on disk.
    args:
        Iterable of command line arguments that should be appended to the
        base command when this input is used.
    metadata:
        Optional free form metadata associated with the input.  This is
        not interpreted by the library but is returned in the
        :class:`BenchmarkResult` for convenience.
    """

    name: str
    args: Tuple[str, ...]
    metadata: Mapping[str, str] | None = None

    @classmethod
    def from_spec(cls, spec: "BenchmarkInput | Mapping[str, object] | Sequence[str] | str") -> "BenchmarkInput":
        """Normalise an arbitrary spec into :class:`BenchmarkInput`.

        The helper accepts a number of convenience formats:

        * ``BenchmarkInput`` instances are returned as-is.
        * ``str`` is treated as both the name and a single argument.
        * ``Sequence[str]`` is interpreted as positional arguments.  The
          first element becomes the name and the full sequence is used as
          command line arguments.
        * ``Mapping`` expects the keys ``name`` and either ``args`` or
          ``value``.
        """

        if isinstance(spec, BenchmarkInput):
            return spec
        if isinstance(spec, str):
            return cls(name=spec, args=(spec,))
        if isinstance(spec, Sequence):
            seq = list(spec)
            if not seq:
                raise ValueError("input sequence specifications must not be empty")
            return cls(name=str(seq[0]), args=tuple(str(part) for part in seq))
        if isinstance(spec, Mapping):
            if "name" not in spec:
                raise ValueError("mapping input specifications must define a 'name'")
            name = str(spec["name"])
            if "args" in spec:
                args = tuple(str(part) for part in spec["args"])  # type: ignore[arg-type]
            elif "value" in spec:
                args = (str(spec["value"]),)
            else:
                args = (name,)
            metadata = None
            meta_value = spec.get("metadata")
            if isinstance(meta_value, Mapping):
                metadata = {str(k): str(v) for k, v in meta_value.items()}
            return cls(name=name, args=args, metadata=metadata)
        raise TypeError(f"Unsupported input specification type: {type(spec)!r}")

    @property
    def slug(self) -> str:
        return _slugify(self.name)


@dataclass
class BenchmarkResult:
    """Outcome of a single Slurm benchmark job."""

    parameters: Dict[str, object]
    input: BenchmarkInput
    job_id: str
    state: str
    run_directory: Path
    script_path: Path
    stdout_path: Path
    stderr_path: Path
    time_log_path: Path
    sacct_records: List[Dict[str, str]]
    time_metrics: Dict[str, str]


class SlurmBenchmarkRunner:
    """Run a parameter sweep using Slurm ``sbatch`` jobs.

    The runner creates a distinct directory per parameter combination and
    input, writes a job script that wraps the user supplied command in
    ``/usr/bin/time --verbose`` and submits the script via ``sbatch``.
    After the job finishes the runner collects metrics from ``sacct`` and
    parses the ``time`` output for convenience.
    """

    def __init__(
        self,
        command: Sequence[str],
        parameter_grid: Mapping[str, Sequence[object]],
        inputs: Iterable[BenchmarkInput | Mapping[str, object] | Sequence[str] | str],
        *,
        workdir: Path | str | None = None,
        base_sbatch_directives: Sequence[str] | None = None,
        resource_option_templates: Mapping[str, str] | None = None,
        sacct_format: Sequence[str] | None = None,
        poll_interval: float = 5.0,
        time_command: str = "/usr/bin/time",
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = tuple(str(part) for part in command)
        self.parameter_grid = {key: list(values) for key, values in parameter_grid.items()}
        if not self.parameter_grid:
            raise ValueError("parameter_grid must contain at least one axis")
        self.inputs = [BenchmarkInput.from_spec(spec) for spec in inputs]
        if not self.inputs:
            raise ValueError("inputs must not be empty")
        self.workdir = Path(workdir) if workdir is not None else Path.cwd() / "slurm_runs"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.base_sbatch_directives = list(base_sbatch_directives or ())
        self.resource_option_templates = dict(_DEFAULT_RESOURCE_TEMPLATES)
        if resource_option_templates:
            self.resource_option_templates.update(resource_option_templates)
        self.sacct_format = list(sacct_format or _DEFAULT_SACCT_FORMAT)
        self.poll_interval = float(poll_interval)
        self.time_command = time_command

    def run(self) -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        combos = list(self._iter_parameter_combinations())
        for index, (parameter_values, benchmark_input) in enumerate(combos, start=1):
            run_dir = self._prepare_run_directory(index, parameter_values, benchmark_input)
            script_path = run_dir / "job.sh"
            stdout_path = run_dir / "command.stdout"
            stderr_path = run_dir / "command.stderr"
            time_log_path = run_dir / "time.log"
            self._write_job_script(
                script_path=script_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                time_log_path=time_log_path,
                parameters=parameter_values,
                benchmark_input=benchmark_input,
            )
            job_id = self._submit_job(script_path)
            state = self._wait_for_completion(job_id)
            sacct_records = self._collect_sacct_data(job_id)
            time_metrics = self._parse_time_log(time_log_path)
            results.append(
                BenchmarkResult(
                    parameters=dict(parameter_values),
                    input=benchmark_input,
                    job_id=job_id,
                    state=state,
                    run_directory=run_dir,
                    script_path=script_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    time_log_path=time_log_path,
                    sacct_records=sacct_records,
                    time_metrics=time_metrics,
                )
            )
        return results

    def _iter_parameter_combinations(self) -> Iterable[Tuple[Dict[str, object], BenchmarkInput]]:
        parameter_names = list(self.parameter_grid.keys())
        value_lists = [self.parameter_grid[name] for name in parameter_names]
        for values in itertools.product(*value_lists):
            combination = dict(zip(parameter_names, values))
            for benchmark_input in self.inputs:
                yield combination, benchmark_input

    def _prepare_run_directory(
        self,
        index: int,
        parameters: Mapping[str, object],
        benchmark_input: BenchmarkInput,
    ) -> Path:
        parameter_slug = "-".join(
            f"{_slugify(str(key))}-{_slugify(str(value))}" for key, value in parameters.items()
        )
        dir_name = f"{index:04d}_{parameter_slug}_{benchmark_input.slug}"
        run_dir = self.workdir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_job_script(
        self,
        *,
        script_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        time_log_path: Path,
        parameters: Mapping[str, object],
        benchmark_input: BenchmarkInput,
    ) -> None:
        sbatch_directives = list(self.base_sbatch_directives)
        sbatch_directives.extend(self._format_parameter_directives(parameters))
        slurm_output = script_path.parent / "slurm-%j.out"
        slurm_error = script_path.parent / "slurm-%j.err"
        sbatch_directives.extend(
            [
                f"--job-name={_slugify(benchmark_input.name)[:200]}",
                f"--output={slurm_output}",
                f"--error={slurm_error}",
            ]
        )
        command_line = shlex.join(self.command + benchmark_input.args)
        time_cmd = shlex.quote(self.time_command)
        script_lines = ["#!/bin/bash"]
        script_lines.extend(f"#SBATCH {directive}" for directive in sbatch_directives)
        script_lines.extend(
            [
                "set -euo pipefail",
                "",
                f"TIME_LOG={shlex.quote(str(time_log_path))}",
                f"CMD_STDOUT={shlex.quote(str(stdout_path))}",
                f"CMD_STDERR={shlex.quote(str(stderr_path))}",
                "",
                "mkdir -p \"$(dirname \"$TIME_LOG\")\"",
                f"{time_cmd} --verbose -o \"$TIME_LOG\" {command_line} > \"$CMD_STDOUT\" 2> \"$CMD_STDERR\"",
            ]
        )
        script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
        script_path.chmod(0o750)

    def _format_parameter_directives(self, parameters: Mapping[str, object]) -> List[str]:
        formatted: List[str] = []
        for name, value in parameters.items():
            template = self.resource_option_templates.get(name, f"--{name}={{value}}").strip()
            if "{value}" not in template:
                template = f"{template}={{value}}"
            formatted.append(template.format(value=value))
        return formatted

    def _submit_job(self, script_path: Path) -> str:
        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("sbatch did not return a job id")
        job_id = stdout.splitlines()[0].strip()
        return job_id

    def _wait_for_completion(self, job_id: str) -> str:
        while True:
            try:
                result = subprocess.run(
                    [
                        "sacct",
                        "-j",
                        job_id,
                        "--format=JobID,State",
                        "--parsable2",
                        "--noheader",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                time.sleep(self.poll_interval)
                continue
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                time.sleep(self.poll_interval)
                continue
            state = self._select_state(job_id, lines)
            if state is None:
                time.sleep(self.poll_interval)
                continue
            simplified = state.split("+", 1)[0]
            if simplified in _ACTIVE_STATES:
                time.sleep(self.poll_interval)
                continue
            return simplified

    def _select_state(self, job_id: str, lines: Sequence[str]) -> Optional[str]:
        fallback: Optional[str] = None
        for line in lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            current_job, state = parts[0], parts[1]
            if current_job == job_id:
                fallback = state
            if current_job.startswith(f"{job_id}.") and current_job.endswith(".batch"):
                return state
        return fallback

    def _collect_sacct_data(self, job_id: str) -> List[Dict[str, str]]:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=" + ",".join(self.sacct_format),
                "--parsable2",
                "--noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        records: List[Dict[str, str]] = []
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        for line in lines:
            parts = line.split("|")
            entry = {column: parts[i] if i < len(parts) else "" for i, column in enumerate(self.sacct_format)}
            records.append(entry)
        return records

    def _parse_time_log(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        metrics: Dict[str, str] = {}
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                metrics[key.strip()] = value.strip()
        return metrics


def run_benchmarks(
    command: Sequence[str],
    parameter_grid: Mapping[str, Sequence[object]],
    inputs: Iterable[BenchmarkInput | Mapping[str, object] | Sequence[str] | str],
    **kwargs: object,
) -> List[BenchmarkResult]:
    """Convenience wrapper around :class:`SlurmBenchmarkRunner`.

    The function instantiates :class:`SlurmBenchmarkRunner` with the
    provided arguments and calls :meth:`SlurmBenchmarkRunner.run`.
    ``kwargs`` are forwarded to the :class:`SlurmBenchmarkRunner`
    constructor.
    """

    runner = SlurmBenchmarkRunner(command, parameter_grid, inputs, **kwargs)
    return runner.run()
