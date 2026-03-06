from pathlib import Path
import subprocess
from typing import Optional, List, Tuple, Dict


def build_pipeline_command(cfg: dict, generated_paths: Dict[str, str]) -> List[str]:
    python_exec = cfg["python_exec"]
    pipeline_script = cfg["pipeline_script"]

    # For now this simply launches pipeline.py as an entrypoint.
    return [python_exec, pipeline_script]


def run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout