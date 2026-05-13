# lib/georef.py

import numpy as np
from .rotations import quat2dcm, R_l2e, T, R1, R2, R3
from pyproj import Transformer
import multiprocessing
from multiprocessing import Pool
import math
import struct
from tqdm import tqdm
from pathlib import Path
import os

cpu_count = multiprocessing.cpu_count()

def loadLasVecAscii(path, cfg):
    """Load laser vectors from CSV file.
    Expected column order: X, Y, Z (SOCS), GPS_time
    Returns ndarray (N, 4): [t_gps, x, y, z]
    """
    import pandas as pd
    from tqdm import tqdm

    delimiter = cfg.get("sep", ",")
    skiprows  = cfg.get("skiprows", 1)
    chunksize = 10_000_000

    chunks = []
    try:
        reader = pd.read_csv(
            path,
            delimiter=delimiter,
            header=None,
            skiprows=skiprows,
            chunksize=chunksize,
            dtype=np.float64,
        )
        file_size = Path(path).stat().st_size
        with tqdm(total=file_size, unit="B", unit_scale=True,
                  desc=f"Loading {Path(path).name}") as pbar:
            prev = 0
            for chunk in reader:
                chunks.append(chunk.values)
                # approximation de la progression par taille lue
                cur = reader.handles.handle.tell() if hasattr(reader, 'handles') else 0
                pbar.update(cur - prev)
                prev = cur
        pbar.n = file_size
        pbar.refresh()
    except Exception as e:
        raise ValueError(f"Cannot open file {path}: {e}")

    data = np.vstack(chunks)
    return data[:, [3, 0, 1, 2, 4]]

def loadLasVecSDC(sdc_file, chunk_records=2_000_000):
    """
    SDC reader with auto-detection of record size.
    Returns:
        ndarray (N, 5): [time, x, y, z, u6]
    """
    sdc_file = Path(sdc_file)

    with open(sdc_file, "rb") as f:
        size_of_header = struct.unpack("<I", f.read(4))[0]

    file_size = os.path.getsize(sdc_file)
    payload_size = file_size - size_of_header

    # --- détection automatique du record size ---
    with open(sdc_file, "rb") as f:
        f.seek(size_of_header)
        probe = f.read(60 * 30)

    record_size = None
    for rs in range(30, 60):
        timestamps = []
        for i in range(20):
            offset = i * rs
            if offset + 8 > len(probe):
                break
            t = struct.unpack("<d", probe[offset:offset+8])[0]
            timestamps.append(t)
        valid = [t for t in timestamps if 40000 < t < 700000]
        if len(valid) >= 18 and payload_size % rs == 0:
            record_size = rs
            break

    if record_size is None:
        # fallback: trouver rs qui donne le plus de timestamps valides même sans divisibilité exacte
        best, best_rs = 0, 40
        for rs in range(30, 60):
            timestamps = [struct.unpack("<d", probe[i*rs:i*rs+8])[0] for i in range(20) if i*rs+8 <= len(probe)]
            valid = sum(1 for t in timestamps if 40000 < t < 700000)
            if valid > best:
                best, best_rs = valid, rs
        record_size = best_rs
        print(f"[SDC] Warning: aucun record_size exact trouvé, utilisation de {record_size} bytes (fallback)")

    print(f"[SDC] record_size détecté = {record_size} bytes")

    record_dtype = np.dtype([
        ("t",   "<f8"),
        ("f1",  "<f4"),
        ("f2",  "<f4"),
        ("x",   "<f4"),
        ("y",   "<f4"),
        ("z",   "<f4"),
        ("u6",  "<u2"),
        ("pad", f"V{record_size - 30}"),
    ])

    assert record_dtype.itemsize == record_size

    record_count = payload_size // record_size
    records = np.empty((record_count, 5), dtype=np.float64)
    write_pos = 0
    total_invalid = 0

    with open(sdc_file, "rb") as f:
        f.seek(size_of_header)
        with tqdm(total=record_count, desc=f"Reading SDC {sdc_file.name}", unit="pts", mininterval=0.5) as pbar:
            while True:
                data = np.fromfile(f, dtype=record_dtype, count=chunk_records)
                n0 = len(data)
                if n0 == 0:
                    break
                valid = (
                    np.isfinite(data["t"]) & np.isfinite(data["x"]) &
                    np.isfinite(data["y"]) & np.isfinite(data["z"])
                )
                n_bad = int(n0 - np.count_nonzero(valid))
                if n_bad > 0:
                    total_invalid += n_bad
                data = data[valid]
                n = len(data)
                if n > 0:
                    records[write_pos:write_pos+n, 0] = data["t"]
                    records[write_pos:write_pos+n, 1] = data["x"]
                    records[write_pos:write_pos+n, 2] = data["y"]
                    records[write_pos:write_pos+n, 3] = data["z"]
                    records[write_pos:write_pos+n, 4] = data["u6"].astype(np.float64)
                    write_pos += n
                pbar.update(n0)

    if total_invalid > 0:
        print(f"[SDC] Dropped {total_invalid} records with non-finite values")

    records = records[:write_pos]

    return records


def _georef_chunk(las_chunk, t_chunk, ecef_chunk, q_chunk, R_sensor2body, lever_arm, ltp_origin=None, lasvec_to_body=False):
    lla2ecef_transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    xyz_body = R_sensor2body @ las_chunk[:, 1:4].T + lever_arm[:, np.newaxis]  
    xyz_ecef = np.zeros_like(xyz_body)

    R_enu2ned = T()
    for i in range(len(las_chunk)):
        R_body2ecef = quat2dcm(q_chunk[i])
        xyz_ecef[:, i] = R_body2ecef @ xyz_body[:, i] + ecef_chunk[i]

    if ltp_origin is not None:
        lat, lon, alt = ltp_origin
        ltp_ecef = np.array(lla2ecef_transformer.transform(lon, lat, alt))
        
        R_enu2e = R_l2e(lat, lon, degrees=True) @ R_enu2ned
        xyz_georef = (R_enu2e.T @ (xyz_ecef - ltp_ecef.reshape(-1, 1))).T
    else:
        ecef2ch1903Transformer = Transformer.from_crs("EPSG:4978", "EPSG:2056")
        xyz_georef = np.array(ecef2ch1903Transformer.transform(xyz_ecef[0, :], xyz_ecef[1, :], xyz_ecef[2, :])).T
    
    # corrections to intensities
    range_m = np.linalg.norm(las_chunk[:, 1:4], axis=1) 
    intensity_raw = las_chunk[:, 4]
    range_clamped = np.clip(range_m, 3.0, np.inf)
    intensity_corrected = las_chunk[:, 4] * range_clamped**(0.5)
    #intensity_corrected = intensity_raw * range_m #**2
    out = np.column_stack((t_chunk, xyz_georef, intensity_corrected))

    if lasvec_to_body:
        v_body = (R_sensor2body @ las_chunk[:, 1:4].T + lever_arm[:, np.newaxis]).T  # lasvec in bodyframe (N, 3)
        out = np.column_stack((out, v_body))  

    return out



def get_R_sensor2body(cfg):
    mount_cfg = cfg['mount']

    if 'R_sensor2body' in mount_cfg:
        return np.array(mount_cfg['R_sensor2body'])
    
    elif 'boresight' in mount_cfg and 'R_mount' in mount_cfg:
        R_mount = np.array(mount_cfg['R_mount'])

        roll = mount_cfg['boresight']['roll']
        pitch = mount_cfg['boresight']['pitch']
        yaw = mount_cfg['boresight']['yaw']

        R_boresight = (R1(roll)@R2(pitch)@R3(yaw)).T
        R_sensor2body = R_boresight @ R_mount # Method LIEO

        return R_sensor2body

    else: 
        raise ValueError("Provide either 'R_sensor2body' or both 'R_mount' and 'boresight' in cfg")


def georefLidar(lasvec, trj, cfg):
    """
    Chunk-level georeferencing with a small Pool.
    Keeps multiprocessing, but only on the current chunk.
    """
    import multiprocessing
    from multiprocessing import Pool
    import numpy as np

    R_sensor2body = get_R_sensor2body(cfg)
    lever_arm = np.array(cfg['mount']['lever_arm'], dtype=np.float64)

    t_interp, ecef_interp, q_interp = trj.interp(lasvec[:, 0], updateSelf=False)

    if 'ltp_origin' in cfg and cfg['ltp_origin'] is not None:
        ltp_origin = np.array(cfg['ltp_origin'])
    else:
        ltp_origin = None

    lasvec_to_body = cfg['output']['lasvec_to_body']

    # -----------------------------
    # compromise: small fixed Pool
    # -----------------------------
    n = len(lasvec)

    # for small chunks, avoid Pool overhead
    if n < 80000:
        return _georef_chunk(
            lasvec,
            t_interp,
            ecef_interp,
            q_interp,
            R_sensor2body,
            lever_arm,
            ltp_origin,
            lasvec_to_body
        )

    n_workers = min(4, multiprocessing.cpu_count(), n)

    las_chunks = np.array_split(lasvec, n_workers)
    t_chunks = np.array_split(t_interp, n_workers)
    ecef_chunks = np.array_split(ecef_interp, n_workers)
    q_chunks = np.array_split(q_interp, n_workers)

    args = [
        (las_c, t_c, ecef_c, q_c, R_sensor2body, lever_arm, ltp_origin, lasvec_to_body)
        for las_c, t_c, ecef_c, q_c in zip(las_chunks, t_chunks, ecef_chunks, q_chunks)
    ]

    with Pool(processes=n_workers) as pool:
        results = pool.starmap(_georef_chunk, args)

    return np.vstack(results)
