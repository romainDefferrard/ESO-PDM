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

def loadLasVecAscii(cfg,limatch_output=False):
    """Load lasver vector from ascii file."""
    try:
        with open(cfg['path'], "r") as f:
            print(f"Loading file {cfg['path']}")
            if 'sep' in cfg and cfg['sep'] is not None:
                data = np.loadtxt(f, delimiter=cfg['sep'])
            else:
                data = np.loadtxt(f)
    except Exception as e:
        errmsg = f" Cannot open file! {str(e)}"
        raise ValueError(errmsg)
    if limatch_output:
        lasvec_A = data[:,[1,5,6,7]]
        lasvec_B = data[:,[0,2,3,4]]
        return np.hstack((lasvec_B, lasvec_A))
    else:
        return data[:, cfg['cols']]

def loadLasVecSDC(sdc_file, chunk_records=2_000_000):
    """
    Fast SDC reader with tqdm progress bar.
    Matches legacy mapping exactly:
      record[[0, 3, 4, 5]]

    Returns:
        ndarray (N, 4): [time, x, y, z]
    """
    sdc_file = Path(sdc_file)

    with open(sdc_file, "rb") as f:
        header_info = f.read(8)
        size_of_header = struct.unpack("<I", header_info[:4])[0]

    record_dtype = np.dtype([
        ("t",    "<f8"),  # field 0
        ("f1",   "<f4"),  # field 1
        ("f2",   "<f4"),  # field 2
        ("x",    "<f4"),  # field 3
        ("y",    "<f4"),  # field 4
        ("z",    "<f4"),  # field 5
        ("u6",   "<u2"),
        ("u7",   "<u2"),
        ("u8",   "u1"),
        ("u9",   "u1"),
        ("u10",  "u1"),
        ("u11",  "<u2"),
        ("u12",  "u1"),
        ("u13",  "u1"),
        ("f14",  "<f4"),
        ("u15",  "<u2"),
    ])

    file_size = os.path.getsize(sdc_file)
    payload_size = file_size - size_of_header
    record_size = record_dtype.itemsize

    if payload_size % record_size != 0:
        raise ValueError(
            f"Invalid SDC size: payload={payload_size} not divisible by record size={record_size}"
        )

    record_count = payload_size // record_size

    # Preallocate final output directly
    records = np.empty((record_count, 4), dtype=np.float64)

    write_pos = 0

    with open(sdc_file, "rb") as f:
        f.seek(size_of_header)

        with tqdm(
            total=record_count,
            desc=f"Reading SDC {sdc_file.name}",
            unit="pts",
            mininterval=0.5,
        ) as pbar:
            while write_pos < record_count:
                data = np.fromfile(f, dtype=record_dtype, count=chunk_records)
                n = len(data)
                if n == 0:
                    break

                records[write_pos:write_pos+n, 0] = data["t"]
                records[write_pos:write_pos+n, 1] = data["x"]
                records[write_pos:write_pos+n, 2] = data["y"]
                records[write_pos:write_pos+n, 3] = data["z"]

                write_pos += n
                pbar.update(n)

    if write_pos != record_count:
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

    out = np.column_stack((t_chunk, xyz_georef))

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
