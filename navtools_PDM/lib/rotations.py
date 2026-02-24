import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

c = np.cos
s = np.sin

def R1(r):
    """
    Rotation matrix around the x-axis.
    :param r: rotation angle in radians
    :return: rotation matrix
    """
    return np.array([[1,    0,    0],
                     [0, c(r), s(r)],
                     [0,-s(r), c(r)]])

def R2(r):
    """
    Rotation matrix around the y-axis.
    :param r: rotation angle in radians
    :return: rotation matrix
    """
    return np.array([[c(r), 0,-s(r)],
                     [   0, 1,    0],
                     [s(r), 0, c(r)]])

def R3(r):
    """
    Rotation matrix around the z-axis.
    :param r: rotation angle in radians
    :return: rotation matrix
    """
    return np.array([[ c(r), s(r), 0],
                     [-s(r), c(r), 0],
                     [    0,    0, 1]])

def R_l2e(lat,lon, degrees=False):
    """
    Rotation matrix from local level NED to ECEF frame.
    :param lat, lon: latitude and longitude in radians
    :return: rotation matrix
    """
    if degrees:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    return np.array([[ -s(lat)*c(lon),-s(lon), -c(lat)*c(lon)],
                     [ -s(lat)*s(lon), c(lon), -c(lat)*s(lon)],
                     [         c(lat),      0,        -s(lat)]])
                     
def T():
    """
    ENU to NED || NED to ENU matrix
    (Also FWD to FLU || FLU to FWD)
    :return: rotation matrix
    """
    return np.array([[ 0, 1, 0],
                     [ 1, 0, 0],
                     [ 0, 0,-1]])

def R1R2R3transp(r, p, y):
    """
    Rotation matrix from rpy angles. Definition follows active rotation as defined in SO polycop.
    Order and transposition follow SBET convention to define rotation from body to local NED (after transposition)

    gives R_b2ned =  (R1(r) @ R2(p) @ R3(y)).T
    """
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    R = np.empty((3, 3))
    R[0, 0] = cp * cy
    R[0, 1] = cy * sp * sr - cr * sy
    R[0, 2] = cr * cy * sp + sr * sy
    R[1, 0] = cp * sy
    R[1, 1] = cr * cy + sp * sr * sy
    R[1, 2] = -cy * sr + cr * sp * sy
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    return R

def dcm2quat(R):
    """
    Convert rotation matrix to quaternion representation
    :param R: rotation matrix
    :return: quaternion [qw, qx, qy, qz]
    """
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return np.array([qw, qx, qy, qz])

def quat2dcm(q):
    """
    Convert a quaternion into a dcm matrix.

    Parameters:
    q : ndarray shape (4,)
        Quaternion in the form [w, x, y, z]

    Returns:
    ndarray shape (3, 3)
        Rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [w*w + x*x - y*y - z*z,     2*(x*y - w*z),         2*(x*z + w*y)],
        [2*(x*y + w*z),             w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
        [2*(x*z - w*y),             2*(y*z + w*x),         w*w - x*x - y*y + z*z]
    ])

def euler2quat(rpy):
    """
    Convert Euler angles to quaternion representation
    :param rpy: Euler angles [roll, pitch, yaw]
    :return: quaternion [qw, qx, qy, qz]
    """
    r, p, y = rpy
    R = R1R2R3transp(r, p, y)
    return dcm2quat(R)

def euler2quat_chunk(chunk):
    return [euler2quat(row) for row in chunk]

def euler2quat_sequence(rpy):
    num_workers = multiprocessing.cpu_count()
    chunk_size = int(np.ceil(len(rpy) / num_workers))
    chunks = [rpy[i:i + chunk_size] for i in range(0, len(rpy), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(euler2quat_chunk, chunks)
    sequence = np.vstack(list(results))
    return sequence

def quat2euler(q):
    """
    q: quaternion [w, x, y, z]
    returns [roll, pitch, yaw] in radians for the Z-Y-X (yaw-pitch-roll) sequence
    """
    w, x, y, z = quatNorm(q)
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    # clamp to handle numerical issues
    sinp_clamped = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp_clamped)

    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)

def quat2euler_chunk(chunk):
    return [quat2euler(row) for row in chunk]

def quat2euler_sequence(q_array):
    q_array = np.atleast_2d(q_array)
    num_workers = multiprocessing.cpu_count()
    chunk_size = int(np.ceil(len(q_array) / num_workers))
    chunks = [q_array[i:i + chunk_size] for i in range(0, len(q_array), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(quat2euler_chunk, chunks)
    sequence = np.vstack(list(results))
    return sequence

def quatInv(q):
    """
    Invert a unit quaternion (i.e. norm 1. If not, q must be normalized it first)
    :param q: quaternion [qw, qx, qy, qz]
    :return: inverted quaternion
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quatMult(q1, q2):
    """
    Compute product of two quaternions.

    Parameters:
    q1, q2 : ndarray shape (4,)
        Quaternions in the form [w, x, y, z]

    Returns:
    ndarray shape (4,)
        Resulting quaternion product [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def quatNorm(q):
    """
    Normalize a quaternion.

    Parameters:
    q : ndarray shape (4,)

    Returns:
    ndarray shape (4,)
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm quaternion cannot be normalized")
    return q / norm

def dcm_to_quat(R):
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    """
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])

