"""Compatibility runner for split_and_shard_subjects on newer Polars."""

import json
import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from MEDS_extract.split_and_shard_subjects.split_and_shard_subjects import shard_subjects
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

CFG_PATH = files("MIMIC_IV_MEDS").joinpath("configs/compat_split_and_shard_subjects.yaml")


@hydra.main(version_base=None, config_path=str(CFG_PATH.parent), config_name=CFG_PATH.stem)
def main(cfg: DictConfig):
    """Run subject splitting without Polars' deprecated streaming parquet path."""

    subsharded_dir = Path(cfg.stage_cfg.data_input_dir)

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(
        f"Reading event conversion config from {event_conversion_cfg_fp} (needed for subject ID columns)"
    )
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    dfs = []

    default_subject_id_col = event_conversion_cfg.pop("subject_id_col", "subject_id")
    for input_prefix, event_cfgs in event_conversion_cfg.items():
        input_subject_id_column = event_cfgs.get("subject_id_col", default_subject_id_col)

        input_fps = list((subsharded_dir / input_prefix).glob("**/*.parquet"))

        input_fps_strs = "\n".join(f"  - {fp.resolve()!s}" for fp in input_fps)
        logger.info(f"Reading subject IDs from {input_prefix} files:\n{input_fps_strs}")

        for input_fp in input_fps:
            dfs.append(
                pl.scan_parquet(input_fp, glob=False)
                .select(pl.col(input_subject_id_column).alias("subject_id"))
                .unique()
            )

    logger.info(f"Joining all subject IDs from {len(dfs)} dataframes")
    subject_ids = (
        pl.concat(dfs, how="vertical_relaxed")
        .select(pl.col("subject_id").drop_nulls().drop_nans().unique())
        # Polars 1.30 dropped the old parquet streaming engine path used by meds-extract 0.3.
        .collect()["subject_id"]
        .to_numpy(use_pyarrow=True)
    )

    logger.info(f"Found {len(subject_ids)} unique subject IDs of type {subject_ids.dtype}")

    if cfg.stage_cfg.external_splits_json_fp:
        external_splits_json_fp = Path(cfg.stage_cfg.external_splits_json_fp)
        if not external_splits_json_fp.exists():
            raise FileNotFoundError(f"External splits JSON file not found at {external_splits_json_fp}")

        logger.info(f"Reading external splits from {external_splits_json_fp.resolve()!s}")
        external_splits = json.loads(external_splits_json_fp.read_text())

        size_strs = ", ".join(f"{k}: {len(v)}" for k, v in external_splits.items())
        logger.info(f"Loaded external splits of size: {size_strs}")
    else:
        external_splits = None

    logger.info("Sharding and splitting subjects")

    sharded_subjects = shard_subjects(
        subjects=subject_ids,
        external_splits=external_splits,
        split_fracs_dict=cfg.stage_cfg.split_fracs,
        n_subjects_per_shard=cfg.stage_cfg.n_subjects_per_shard,
        seed=cfg.seed,
    )

    shards_map_fp = Path(cfg.shards_map_fp)
    logger.info(f"Writing sharded subjects to {shards_map_fp.resolve()!s}")
    shards_map_fp.parent.mkdir(parents=True, exist_ok=True)
    shards_map_fp.write_text(json.dumps(sharded_subjects))
    logger.info("Done writing sharded subjects")


if __name__ == "__main__":
    main()
