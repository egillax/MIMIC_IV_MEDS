#!/usr/bin/env python

import logging
import shlex
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from . import ETL_CFG, EVENT_CFG, HAS_PRE_MEDS, MAIN_CFG
from . import __version__ as PKG_VERSION
from . import dataset_info
from .commands import run_command
from .download import download_data

if HAS_PRE_MEDS:
    from .pre_MEDS import main as pre_MEDS_transform

logger = logging.getLogger(__name__)


def get_transform_bin_dir() -> Path:
    """Return the scripts directory for the current Python environment."""
    return Path(sys.executable).parent


def get_pipeline_command() -> str:
    """Return a pipeline command bound to the current Python environment."""
    pipeline_executable = get_transform_bin_dir() / "MEDS_transform-pipeline"
    return shlex.quote(str(pipeline_executable))


def prepare_stage_runner_config(stage_runner_fp: str | None, root_output_dir: Path) -> Path:
    """Write a stage runner config that routes split_and_shard_subjects through a compat shim."""
    compat_script = f"{shlex.quote(sys.executable)} -m MIMIC_IV_MEDS.compat.split_and_shard_subjects"
    compat_cfg = OmegaConf.create({"split_and_shard_subjects": {"script": compat_script}})

    if stage_runner_fp:
        merged_cfg = OmegaConf.merge(OmegaConf.load(stage_runner_fp), compat_cfg)
    else:
        merged_cfg = compat_cfg

    compat_stage_runner_fp = root_output_dir / ".compat_stage_runner.yaml"
    compat_stage_runner_fp.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged_cfg, compat_stage_runner_fp)
    return compat_stage_runner_fp


@hydra.main(version_base=None, config_path=str(MAIN_CFG.parent), config_name=MAIN_CFG.stem)
def main(cfg: DictConfig):
    """Runs the end-to-end MEDS Extraction pipeline."""

    raw_input_dir = Path(cfg.raw_input_dir)
    pre_MEDS_dir = Path(cfg.pre_MEDS_dir)
    MEDS_output_dir = Path(cfg.MEDS_output_dir)
    stage_runner_fp = prepare_stage_runner_config(cfg.get("stage_runner_fp", None), raw_input_dir.parent)

    # Step 0: Data downloading
    if cfg.do_download:
        if cfg.get("do_demo", False):
            logger.info("Downloading demo data.")
            download_data(raw_input_dir, dataset_info, do_demo=True)
        else:
            logger.info("Downloading data.")
            download_data(raw_input_dir, dataset_info)
    else:  # pragma: no cover
        logger.info("Skipping data download.")

    # Step 1: Pre-MEDS Data Wrangling
    if HAS_PRE_MEDS:
        pre_MEDS_transform(
            input_dir=raw_input_dir,
            output_dir=pre_MEDS_dir,
            do_overwrite=cfg.get("do_overwrite", None),
            do_copy=cfg.get("do_copy", None),
        )
    else:
        pre_MEDS_dir = raw_input_dir

    # Step 2: MEDS Cohort Creation
    # First we need to set some environment variables
    command_parts = [
        f"DATASET_NAME={dataset_info.dataset_name}",
        f"DATASET_VERSION={dataset_info.raw_dataset_version}:{PKG_VERSION}",
        f"EVENT_CONVERSION_CONFIG_FP={str(EVENT_CFG.resolve())}",
        f"PRE_MEDS_DIR={str(pre_MEDS_dir.resolve())}",
        f"MEDS_OUTPUT_DIR={str(MEDS_output_dir.resolve())}",
        f"PATH={shlex.quote(str(get_transform_bin_dir()))}:$PATH",
    ]

    command_parts.append(get_pipeline_command())
    command_parts.append(f"pipeline_config_fp={str(ETL_CFG.resolve())}")

    command_parts.append(f"stage_runner_fp={str(stage_runner_fp.resolve())}")
    if cfg.get("do_profile", False):
        command_parts.append("do_profile=True")

    run_command(command_parts)


if __name__ == "__main__":
    main()
