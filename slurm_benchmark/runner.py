"""Barebones Slurm benchmarking utilities."""
from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Mapping, MutableMapping, Sequence

import pandas as pd


_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


def _normalize_option(option: str) -> str:
    if option.startswith("--"):
        return option
    if option.startswith("-"):
        return f"-{option.lstrip('-')}"
    return f"--{option}"


@dataclass
class SlurmBenchmark:
    """Coordinate benchmarking jobs on a Slurm cluster.

    Parameters
    ----------
    command_template:
        Sequence of command arguments where Python ``str.format`` placeholders are
        allowed for parameter substitution. Example::

            ["python", "train.py", "--k", "{k}"]

    parameters:
        Mapping of parameter names to the list of values that should be explored.
        These values will be injected into the command template by name.
    resources:
        Mapping of sbatch resource flags (``--mem``, ``--cpus-per-task``, …) to the
        values that should be requested for each benchmark run.
    job_name:
        Base job name supplied to sbatch. The actual job name is suffixed with an
        index to keep the names unique.
    delay:
        Optional delay (in seconds) to apply between job submissions in order to
        avoid overwhelming the scheduler.
    sacct_format:
        Value for the ``--format`` option passed to sacct when gathering job
        statistics.
    sacct_path:
        Override the executable used for sacct. Defaults to ``sacct`` which must be
        available in ``PATH`` when the library is used.
    sbatch_path:
        Override the executable used for sbatch. Defaults to ``sbatch`` which must be
        available in ``PATH`` when the library is used.
    dryrun:
        When ``True``, commands are only printed instead of submitted to Slurm and no
        statistics are collected.
    squeue_path:
        Override the executable used for squeue when waiting for jobs to finish.
    squeue_poll_interval:
        Delay (in seconds) between squeue polling attempts while waiting for jobs to
        complete.
    """

    command_template: Sequence[str]
    parameters: Mapping[str, Sequence[object]]
    resources: Mapping[str, Sequence[object]]
    job_name: str = "slurm-benchmark"
    delay: float = 0.0
    sacct_format: str = "JobID,JobName,State,Elapsed,MaxRSS,TotalCPU"
    sacct_path: str = "sacct"
    sbatch_path: str = "sbatch"
    extra_sacct_args: Sequence[str] = field(default_factory=list)
    extra_sbatch_args: Sequence[str] = field(default_factory=list)
    dryrun: bool = False
    squeue_path: str = "squeue"
    squeue_poll_interval: float = 5.0

    def run(self) -> pd.DataFrame:
        """Submit the benchmarking jobs and collect the sacct output."""
        job_records: List[MutableMapping[str, object]] = []

        parameter_keys = list(self.parameters.keys())
        resource_keys = list(self.resources.keys())

        param_values = [self.parameters[key] for key in parameter_keys]
        resource_values = [self.resources[key] for key in resource_keys]

        param_combinations = list(product(*param_values)) if param_values else [tuple()]
        resource_combinations = (
            list(product(*resource_values)) if resource_values else [tuple()]
        )

        for index, (param_combo, resource_combo) in enumerate(
            product(param_combinations, resource_combinations)
        ):
            params = dict(zip(parameter_keys, param_combo))
            resources = dict(zip(resource_keys, resource_combo))
            record = self._submit_job(index, params, resources)
            job_records.append(record)
            if self.delay:
                time.sleep(self.delay)

        if not self.dryrun:
            sacct_records = self._collect_sacct(job_records)

            for record, sacct_record in zip(job_records, sacct_records):
                record.update(
                    {f"sacct_{key}": value for key, value in sacct_record.items()}
                )

        return pd.DataFrame(job_records)

    # ------------------------------------------------------------------
    def _submit_job(
        self, index: int, params: Mapping[str, object], resources: Mapping[str, object]
    ) -> MutableMapping[str, object]:
        command_context: Dict[str, object] = dict(params)
        formatted_command = [
            str(part).format(**command_context) for part in self.command_template
        ]
        wrapped_command = " ".join(shlex.quote(arg) for arg in formatted_command)

        sbatch_cmd: List[str] = [self.sbatch_path]
        sbatch_cmd.extend(self.extra_sbatch_args)
        job_name = f"{self.job_name}-{index}"
        sbatch_cmd.extend(["--job-name", job_name])
        for option, value in resources.items():
            normalized_option = _normalize_option(option)
            sbatch_cmd.extend([normalized_option, str(value)])
        sbatch_cmd.extend(["--wrap", wrapped_command])

        record: MutableMapping[str, object] = {
            "job_id": None,
            "job_name": job_name,
            "params": params,
            "resources": resources,
            "sbatch_command": sbatch_cmd,
            "wrapped_command": wrapped_command,
        }

        if self.dryrun:
            print(" ".join(shlex.quote(part) for part in sbatch_cmd))
            return record

        result = subprocess.run(
            sbatch_cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        job_id_match = _JOB_ID_PATTERN.search(result.stdout)
        if not job_id_match:
            raise RuntimeError(
                "Failed to extract job ID from sbatch output: " f"{result.stdout!r}"
            )

        job_id = job_id_match.group(1)

        record["job_id"] = job_id
        return record

    # ------------------------------------------------------------------
    def _collect_sacct(self, job_records: Sequence[Mapping[str, object]]) -> List[Dict[str, str]]:
        sacct_records: List[Dict[str, str]] = []
        for record in job_records:
            job_id = str(record["job_id"])
            self._wait_for_job_completion(job_id)
            sacct_cmd = [
                self.sacct_path,
                "-j",
                job_id,
                "--format",
                self.sacct_format,
                "--parsable2",
                "--noheader",
                *self.extra_sacct_args,
            ]
            result = subprocess.run(
                sacct_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            sacct_records.append(self._parse_sacct_output(result.stdout))
        return sacct_records

    # ------------------------------------------------------------------
    def _wait_for_job_completion(self, job_id: str) -> None:
        while True:
            result = subprocess.run(
                [self.squeue_path, "-j", job_id, "-h"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to query job status via squeue: " f"{result.stderr!r}"
                )
            if not result.stdout.strip():
                break
            time.sleep(self.squeue_poll_interval)

    # ------------------------------------------------------------------
    def _parse_sacct_output(self, output: str) -> Dict[str, str]:
        output = output.strip()
        if not output:
            return {}
        first_line = output.splitlines()[0]
        fields = self.sacct_format.split(",")
        values = first_line.split("|")
        values = values[: len(fields)]
        return dict(zip(fields, values))
