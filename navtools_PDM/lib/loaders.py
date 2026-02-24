import os
import shutil
import tarfile
import numpy as np
import pyproj
from datetime import datetime
import matplotlib.pyplot as plt


proj_ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
proj_lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

from pyproj import Transformer

lla2ecefTransformer = Transformer.from_crs("EPSG:4326", "EPSG:4978")
ecef2llaTransformer = Transformer.from_crs("EPSG:4978", "EPSG:4326")


def loadSBET(path,cfg):
    """
    Decodes an APPLANIX SNV/SBET file.

    Parameters:
    - settings: path to SBET

    Returns:
    - data: numpy array of processed data

  Input record: 17xdouble=(136 bytes)
       0  time  			sec_of_week 
       1  latitude   		rad
       2  longitude  		rad
       3  altitude       meters
       4  x_wander_vel   m/s
       5  y_wander_vel   m/s
       6  z_wander_vel  	m/s
       7  roll          	radians
       8  pitch         	radians
       9  wander_heading radians
       10 wander angle   radians
       11 x body accel   m/s^2
       12 y body accel   m/s^2
       13 z body accel   m/s^2
       14 x angular rate rad/s
       15 y angular rate rad/s
	   16 z angular rate rad/s					
 This is what is written in the ouput record:
       0   time            sec_of_week
       1   latitude        rad
       2   longitude       rad
       3   altitude        m
       4   roll            rad
       5   pitch           rad
       6  heading         rad 
    """

    try:
        with open(path, "rb") as f:
            print(f"Loading file {path}")
            data = np.fromfile(f, dtype=np.float64).reshape(-1,17)
    except Exception as e:
        errmsg = f"Cannot open file! {str(e)}"
        raise ValueError(errmsg)
        
    time = data[:, 0]
    if 'timeRef' in cfg and cfg['timeRef'] == 'UTC':
        print(" Converting time from UTC to GPS sow")
        time += 18

    lla = data[:, 1:4]
    x,y,z = lla2ecefTransformer.transform(lla[:, 0], lla[:, 1], lla[:, 2],radians=True)
    ecef = np.dstack((x, y, z))[0]
    rpy = data[:, 7:9]
    rpy = np.column_stack((rpy, data[:, 9]-data[:, 10]))

    return time, lla, ecef, rpy

def writeSBET(path, time, lla, rpy):
    """
    Write data to an APPLANIX SBET file.

    Parameters:
    time: numpy array (N), gps seconds of week
    lla: numpy array of shape (N, 3) [lat (rad), lon (rad), alt (m)]
    rpy: numpy array of shape (N, 3) [roll (rad), pitch (rad), heading (rad)]

    What is written in the ouput record:
       0  time  			sec_of_week 
       1  latitude   		rad
       2  longitude  		rad
       3  altitude       meters
       4  x_wander_vel   m/s
       5  y_wander_vel   m/s
       6  z_wander_vel  	m/s
       7  roll          	radians
       8  pitch         	radians
       9  wander_heading radians
       10 wander angle   radians
       11 x body accel   m/s^2
       12 y body accel   m/s^2
       13 z body accel   m/s^2
       14 x angular rate rad/s
       15 y angular rate rad/s
	   16 z angular rate rad/s	
    """

    n = len(time)
    data = np.zeros((n,17), dtype=np.float64)
    data[:, 0] = time
    data[:, 1:4] = lla
    data[:, 7:10] = rpy
  
    try:
        with open(path, "wb") as f:
            print(f"Writing SBET file to {path}")
            data.tofile(f)
    except Exception as e:
        errmsg = f" Cannot write file! {str(e)}"
        raise ValueError(errmsg)


def loadASCII(path, cfg):
    """
    Load ASCII file.
    """
    try:
        with open(path, "r") as f:
            print(f"Loading file {path}")
            if 'delimiter' in cfg and cfg['delimiter'] is not None:
                data = np.loadtxt(f, delimiter=cfg['delimiter'],skiprows=cfg['header'])
            else:
                data = np.loadtxt(f, skiprows=cfg['header'])
    except Exception as e:
        errmsg = f" Cannot open file! {str(e)}"
        raise ValueError(errmsg)

    
    time = data[:, cfg['t_col']]
    if 'timeRef' in cfg and cfg['timeRef'] == 'UTC':
        print(" Converting time from UTC to GPS sow")
        time += 18
    
    if 'lla_col' in cfg and cfg['lla_col'] is not None:
        
        lla = data[:, cfg['lla_col']]
        llRad = ('llaUnit' in cfg and cfg['llaUnit'] == 'rad')
        x,y,z = lla2ecefTransformer.transform(lla[:, 0], lla[:, 1], lla[:, 2], radians=llRad)
        ecef = np.dstack((x, y, z))[0]

        print("  Using LLA data to compute ECEF")
        print(f"  LLA data in {'radians' if llRad else 'degrees'}")

    elif 'ecef_col' in cfg and cfg['ecef_col'] is not None:
        print("  Using ECEF data to compute LLA")
        ecef = data[:, cfg['ecef_col']]
        lon, lat, alt = ecef2llaTransformer.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2], radians=True)
        lla = np.dstack((lon, lat, alt))[0]
    else:
        print("  No LLA or ECEF data available, setting to None")
        lla = None
        ecef = None
        

    rpy = data[:, cfg['rpy_col']] if cfg['rpy_col'] is not None else None
    q = data[:, cfg['q_col']] if cfg['q_col'] is not None else None


    if cfg['rpyUnit'] == 'deg':
        print("  Converting RPY from degrees to radians")
        rpy = np.deg2rad(rpy)
    elif cfg['rpyUnit'] == 'rad':
        print("  RPY interpreted in radians")
        pass
    
    #assert that time is strictly increasing
    #check duplicates and delete 
    unique_time, unique_indices = np.unique(time, return_index=True)
    if len(unique_time) < len(time):
        print(f" Warning: Found {len(time)-len(unique_time)} duplicate time entries. Removing duplicates.")
        time = unique_time
        if lla is not None:
            lla = lla[unique_indices]
        if ecef is not None:
            ecef = ecef[unique_indices]
        if rpy is not None:
            rpy = rpy[unique_indices]
        if q is not None:
            q = q[unique_indices]
    if not np.all(np.diff(time) > 0):
        print(" Warning: Time vector is not strictly increasing. Sorting data by time.")
        sorted_indices = np.argsort(time)
        time = time[sorted_indices]
        if lla is not None:
            lla = lla[sorted_indices]
        if ecef is not None:
            ecef = ecef[sorted_indices]
        if rpy is not None:
            rpy = rpy[sorted_indices]
        if q is not None:
            q = q[sorted_indices]
    return time, lla, ecef, rpy, q

def loadDN(path, cfg):
    """
    Load ROAMFREE trajectory data from a .log file, possibly inside a .tar.gz archive.

    Parameters:
    - path (str): Path to PoseSE3(W).log file or .tar.gz archive containing it.
    - config (dict): Configuration options, can include:
        - 'downSample': int, sample every n-th entry
        - 'timeOffset': float, offset to apply to timestamps

    Returns:
    - t: numpy array of timestamps
    - lla: None (not available)
    - ecef: numpy array of ECEF positions
    - rpy: None (not available)
    - q: numpy array of quaternions
    """

    if path.endswith('.tar.gz'):
        print(f"Loading file {path}")
        timestamp = datetime.now().strftime('%Y%m%d')
        tmp_path = os.path.join('/tmp', f'dn_{timestamp}')

        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)

        os.makedirs(tmp_path, exist_ok=True)
        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(path=tmp_path)

        path = tmp_path

    log_path = os.path.join(path, 'PoseSE3(W).log')

    try:
        raw = np.loadtxt(log_path,delimiter=',')
    except Exception as e:
        raise ValueError(f"Failed to load log file at {log_path}: {e}")


    if 'downSample' in cfg and cfg['downSample'] > 1:
        raw = raw[::cfg['downSample'], :]
    
    sorted_indices = np.argsort(raw[:, 0])
    raw = raw[sorted_indices]

    time = raw[:, 0]
    if 'timeRef' in cfg and cfg['timeRef'] == 'UTC':
        print(" Converting time from UTC to GPS sow")
        time += 18

    ecef = raw[:, 2:5] + cfg['originShift']
    lon,lat,alt = ecef2llaTransformer.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2], radians=True)
    lla = np.dstack((lon,lat,alt))[0]
    q = raw[:, 5:9]


    return time, ecef, lla, q