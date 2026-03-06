from pathlib import Path
import shutil
import yaml

from .pointCloudGeoref import run_from_yaml
from .singleBeamMerging import merge_txt_pairs
from .Chunker import write_gps_multi_outage
from .gnss_scenarios import write_gps_cycle_slips

from .pipeline import (
    chunk_txt,
    combined_multi_outage_scenario,
    run_limatch_on_chunks_per_scan,
    run_patcher_cli,
    run_limatch_on_patcher_outputs,
)

from .sync_configs import sync_all_configs


def _safe_load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _get(cfg, *keys, **kwargs):
    default = kwargs.get("default", None)
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def georef_and_merge_app(
    cfg_ha_path,
    cfg_lr_path,
    merged_out,
    delete_tmp_after_success=False,
    delimiter=",",
    skiprows=0,
    sort_by_time=True,
    out_prefix="merged_",
    out_suffix="_HA_LR",
):
    cfg_ha = _safe_load_yaml(cfg_ha_path)
    cfg_lr = _safe_load_yaml(cfg_lr_path)

    dir_ha = Path(cfg_ha["output"]["path"])
    dir_lr = Path(cfg_lr["output"]["path"])
    out_dir = Path(merged_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_from_yaml(str(cfg_ha_path))
    run_from_yaml(str(cfg_lr_path))

    merge_txt_pairs(
        dir_a=dir_ha,
        dir_b=dir_lr,
        out_dir=out_dir,
        delimiter=delimiter,
        skiprows=skiprows,
        sort_by_time=sort_by_time,
        out_prefix=out_prefix,
        out_suffix=out_suffix,
    )

    merged_files = list(out_dir.glob("*.txt"))
    if len(merged_files) == 0:
        raise RuntimeError("Merge produced no .txt files.")

    deleted = False
    if delete_tmp_after_success:
        tmp_root = dir_ha.parents[1]
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
            deleted = True

    return {
        "merged_dir": str(out_dir),
        "merged_n": len(merged_files),
        "deleted_tmp": deleted,
    }


def run_mode_georef_only(app_cfg, sync_result):
    return georef_and_merge_app(
        cfg_ha_path=sync_result["ha_yaml"],
        cfg_lr_path=sync_result["lr_yaml"],
        merged_out=sync_result["merged_dir"],
        delete_tmp_after_success=_get(app_cfg, "execution", "delete_tmp_after_success", default=False),
        delimiter=_get(app_cfg, "merge", "delimiter", default=","),
        skiprows=int(_get(app_cfg, "merge", "skiprows", default=0)),
        sort_by_time=bool(_get(app_cfg, "merge", "sort_by_time", default=True)),
        out_prefix=_get(app_cfg, "merge", "out_prefix", default="merged_"),
        out_suffix=_get(app_cfg, "merge", "out_suffix", default="_HA_LR"),
    )


def run_mode_chunk(app_cfg, sync_result):
    if _get(app_cfg, "execution", "do_georef_merge", default=True):
        georef_and_merge_app(
            cfg_ha_path=sync_result["ha_yaml"],
            cfg_lr_path=sync_result["lr_yaml"],
            merged_out=sync_result["merged_dir"],
            delete_tmp_after_success=_get(app_cfg, "execution", "delete_tmp_after_success", default=False),
            delimiter=_get(app_cfg, "merge", "delimiter", default=","),
            skiprows=int(_get(app_cfg, "merge", "skiprows", default=0)),
            sort_by_time=bool(_get(app_cfg, "merge", "sort_by_time", default=True)),
            out_prefix=_get(app_cfg, "merge", "out_prefix", default="merged_"),
            out_suffix=_get(app_cfg, "merge", "out_suffix", default="_HA_LR"),
        )

    chunks_root = chunk_txt(
        merged_dir=sync_result["merged_dir"],
        cfg_georef_path=str(sync_result["ha_yaml"]),
        chunks_out=str(sync_result["chunks_dir"]),
        L=float(_get(app_cfg, "chunk", "length_m", default=15.0)),
        epsg_out=_get(app_cfg, "chunk", "epsg_out", default="EPSG:2056"),
        delimiter=_get(app_cfg, "merge", "delimiter", default=","),
        skiprows=int(_get(app_cfg, "merge", "skiprows", default=0)),
        min_points=int(_get(app_cfg, "chunk", "min_points", default=2000)),
    )

    limatch_out = sync_result["limatch_dir"]

    run_limatch_on_chunks_per_scan(
        chunks_root=chunks_root,
        limatch_cfg=Path(sync_result["limatch_yaml"]),
        out_root=limatch_out,
        do_cross_scan=bool(_get(app_cfg, "chunk", "do_cross_scan", default=True)),
        neighbor_k=int(_get(app_cfg, "chunk", "neighbor_k", default=1)),
    )

    return {
        "merged_dir": str(sync_result["merged_dir"]),
        "chunks_root": str(chunks_root),
        "limatch_out": str(limatch_out),
    }


def run_mode_outage_chunk(app_cfg, sync_result):
    if _get(app_cfg, "execution", "do_georef_merge", default=True):
        georef_and_merge_app(
            cfg_ha_path=sync_result["ha_yaml"],
            cfg_lr_path=sync_result["lr_yaml"],
            merged_out=sync_result["merged_dir"],
            delete_tmp_after_success=_get(app_cfg, "execution", "delete_tmp_after_success", default=False),
            delimiter=_get(app_cfg, "merge", "delimiter", default=","),
            skiprows=int(_get(app_cfg, "merge", "skiprows", default=0)),
            sort_by_time=bool(_get(app_cfg, "merge", "sort_by_time", default=True)),
            out_prefix=_get(app_cfg, "merge", "out_prefix", default="merged_"),
            out_suffix=_get(app_cfg, "merge", "out_suffix", default="_HA_LR"),
        )

    outages = _get(app_cfg, "outage", "outages", default=None)
    if outages is None:
        outages = [
            [
                float(_get(app_cfg, "outage", "start", default=0.0)),
                float(_get(app_cfg, "outage", "duration", default=0.0)),
            ]
        ]

    chunks_root, gps_outage = combined_multi_outage_scenario(
        merged_dir=sync_result["merged_dir"],
        cfg_georef_path=sync_result["ha_yaml"],
        gps_in=Path(app_cfg["paths"]["gps_in"]),
        outages=outages,
        pre=float(_get(app_cfg, "outage", "pre", default=30.0)),
        post=float(_get(app_cfg, "outage", "post", default=30.0)),
        out_root=sync_result["scenario_combined_dir"],
        delimiter=_get(app_cfg, "merge", "delimiter", default=","),
        min_points_chunk=int(_get(app_cfg, "chunk", "min_points", default=2000)),
        epsg_out=_get(app_cfg, "chunk", "epsg_out", default="EPSG:2056"),
        do_chunks=bool(_get(app_cfg, "execution", "do_chunks", default=True)),
        reuse_chunks=bool(_get(app_cfg, "execution", "reuse_chunks", default=True)),
        force=bool(_get(app_cfg, "execution", "force_rebuild_chunks", default=False)),
    )

    limatch_out = sync_result["scenario_combined_dir"] / "limatch"

    run_limatch_on_chunks_per_scan(
        chunks_root=chunks_root,
        limatch_cfg=Path(sync_result["limatch_yaml"]),
        out_root=limatch_out,
        do_cross_scan=bool(_get(app_cfg, "chunk", "do_cross_scan", default=True)),
        neighbor_k=int(_get(app_cfg, "chunk", "neighbor_k", default=1)),
    )

    return {
        "merged_dir": str(sync_result["merged_dir"]),
        "chunks_root": str(chunks_root),
        "gps_outage": str(gps_outage),
        "limatch_out": str(limatch_out),
    }


def run_mode_patcher(app_cfg, sync_result):
    run_patcher_cli(Path(sync_result["patcher_yaml"]))
    return {
        "patcher_cfg": str(sync_result["patcher_yaml"]),
        "patcher_out": str(sync_result["patcher_dir"]),
    }


def run_mode_patcher_limatch(app_cfg, sync_result):
    patcher_out_root = Path(app_cfg["paths"]["patcher_out_root"])

    gps_out = None
    if _get(app_cfg, "execution", "do_gps_outage", default=False):
        outages = _get(app_cfg, "outage", "outages", default=None)
        if outages is None:
            outages = [
                [
                    float(_get(app_cfg, "outage", "start", default=0.0)),
                    float(_get(app_cfg, "outage", "duration", default=0.0)),
                ]
            ]

        out_dir = sync_result["scenario_gps_outage_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        gps_out = out_dir / "GPS.txt"

        write_gps_multi_outage(
            gps_in=Path(app_cfg["paths"]["gps_in"]),
            gps_out=gps_out,
            outages=outages,
            delimiter=_get(app_cfg, "merge", "delimiter", default=","),
        )

    run_limatch_on_patcher_outputs(
        patcher_out_root=patcher_out_root,
        limatch_cfg=Path(sync_result["limatch_yaml"]),
        out_root=sync_result["limatch_dir"],
    )

    return {
        "patcher_out_root": str(patcher_out_root),
        "limatch_out": str(sync_result["limatch_dir"]),
        "gps_out": str(gps_out) if gps_out is not None else None,
    }


def run_mode_cycleslip(app_cfg, sync_result):
    out_root = sync_result["scenario_root"] / "cycle_slip"
    out_root.mkdir(parents=True, exist_ok=True)

    scenarios = _get(app_cfg, "cycleslip", "scenarios", default=None)
    written = []

    for sc in scenarios:
        sc_dir = out_root / sc["name"]
        sc_dir.mkdir(parents=True, exist_ok=True)
        gps_out = sc_dir / "GPS.txt"

        write_gps_cycle_slips(
            gps_in=Path(app_cfg["paths"]["gps_in"]),
            gps_out=gps_out,
            slips=sc["slips"],
            delimiter=_get(app_cfg, "merge", "delimiter", default=","),
        )
        written.append(gps_out)

    return {
        "cycle_slip_root": str(out_root),
        "n_scenarios": len(written),
    }


def run_pipeline_app(app_cfg, auto_sync=True):
    mode = app_cfg["scenario"]["mode"]

    sync_result = sync_all_configs(app_cfg) if auto_sync else sync_all_configs(app_cfg)

    if mode == "GeorefOnly":
        return run_mode_georef_only(app_cfg, sync_result)

    if mode == "Chunk":
        return run_mode_chunk(app_cfg, sync_result)

    if mode == "OutageChunk":
        return run_mode_outage_chunk(app_cfg, sync_result)

    if mode == "Patcher":
        return run_mode_patcher(app_cfg, sync_result)

    if mode == "PatcherLiMatch":
        return run_mode_patcher_limatch(app_cfg, sync_result)

    if mode == "CycleSlip":
        return run_mode_cycleslip(app_cfg, sync_result)

    if mode in [None, "None"]:
        return {"status": "nothing_to_run"}

    raise ValueError("Unsupported mode: {0}".format(mode))