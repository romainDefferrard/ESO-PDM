"""
python -m pipeline -c cfg/pipeline_outage_1.yml
"""
from pipeline.pipeline import run_pipeline
import argparse
from pathlib import Path
import yaml

parser = argparse.ArgumentParser(description="MLS processing pipeline")
parser.add_argument("-c", "--config", required=True)
args = parser.parse_args()

cfg_path = Path(args.config).resolve()
with open(cfg_path) as f:
    pipe_cfg = yaml.safe_load(f)

# Inject config dir so relative scanner paths resolve correctly
pipe_cfg["_cfg_dir"] = str(cfg_path.parent)

run_pipeline(pipe_cfg)
