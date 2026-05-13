import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .loaders import loadSBET, loadASCII, loadDN
from .rotations import *
from cycler import cycler
from scipy.spatial.transform import Rotation as R, Slerp
import multiprocessing

epfl_colors = [
    "#007480",  # Canard
    "#B51F1F",  # Groseille
    "#413D3A",  # Ardoise
    "#00A79F",  # Léman
    "#FF0000",  # Rouge
    "#CAC7C7",  # Perle
]

mpl.rcParams['axes.formatter.use_mathtext'] = True


plt.rcParams['axes.prop_cycle'] = cycler(color=epfl_colors)
plt.rcParams.update({
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'font.size': 12,
    'font.family':  ('cmr10', 'STIXGeneral'),
    'lines.linewidth': 0.75,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
})

def _compute_q_l2e_chunk(chunk):
    """Compute local→ECEF quaternions for a chunk of LLA rows."""
     
    out = []
    for lat, lon, _ in chunk: 
        R = R_l2e(lat, lon)
        q = dcm2quat(R)
        out.append(q)
    return np.array(out)

def _slerp_chunk(args):
    """Interpolate a chunk of times using SciPy Slerp, converting quat order."""
    t_chunk, time, q_repo = args
    q_scipy = np.column_stack((q_repo[:, 1:], q_repo[:, 0]))
    key_rots = R.from_quat(q_scipy)
    slerp = Slerp(time, key_rots)
    interp_rots = slerp(t_chunk)
    q_interp_scipy = interp_rots.as_quat()
    
    return np.column_stack((q_interp_scipy[:, 3], q_interp_scipy[:, :3]))

class Trajectory:
    """
    A simple trajectory class.
    Each array must be of same length and synchronized!
    Stores time and optional spatial representations.
    """
    def __init__(self, time, lla=None, ecef=None, q=None, cfg=None):
        self.time = time
        self.lla = lla
        self.ecef = ecef
        self.q = q
        self.cfg = cfg

        if lla is not None:
            assert len(time) == len(lla), "Time and LLA arrays must be of same length"
        if ecef is not None:
            assert len(time) == len(ecef), "Time and ECEF arrays must be of same length"
        if q is not None:
            assert len(time) == len(q), "Time and Q arrays must be of same length"

    @classmethod
    def fromSBET(cls, path, cfg):
        """
        Instanciate Trajectory from SBET file.
        """
        time, lla, ecef, rpy = loadSBET(path,cfg)

        q = euler2quat_sequence(rpy)

        return cls(
            time=time.astype(np.float64),
            lla=lla,
            ecef=ecef,
            q=q,
            cfg=cfg
        )
    
    @classmethod
    def fromASCII(cls, path, cfg):
        """
        Instanciate Trajectory from ASCII file.
        """
        time, lla, ecef, rpy, q = loadASCII(path, cfg)

        if q is None:
            q = euler2quat_sequence(rpy)

        return cls(
            time=time.astype(np.float64),
            lla=lla,
            ecef=ecef,
            q=q,
            cfg=cfg
        )
    
    @classmethod
    def fromDN(cls, path, cfg):
        """
        Instanciate Trajectory from ROAMFREE DN file.
        """
        time, ecef, lla, q = loadDN(path, cfg)

        return cls(
            time=time.astype(np.float64),
            ecef=ecef,
            lla=lla,
            q=q,
            cfg=cfg
        )
    
 
    def interp(self, t_interp, updateSelf =True):
        """
        Interpolate trajectory at given time points.
        Positions/LLA: linear interp (parallelized)
        Orientations: slerp on SO(3) in parallel chunks,
                      converting between [w,x,y,z] (repo) <-> [x,y,z,w] (SciPy).
        """
        t_interp = np.asarray(t_interp)

        ecef_interp = np.zeros((len(t_interp), 3))
        for i in range(3):
            ecef_interp[:, i] = np.interp(t_interp, self.time, self.ecef[:, i])
        lla_interp = np.zeros((len(t_interp), 3))
        for i in range(3):
            lla_interp[:, i] = np.interp(t_interp, self.time, self.lla[:, i])

        num_workers = multiprocessing.cpu_count() - 1
        chunk_size = int(np.ceil(len(t_interp) / num_workers))
        chunks = [t_interp[i:i + chunk_size] for i in range(0, len(t_interp), chunk_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = ex.map(_slerp_chunk, [(chunk, self.time, self.q) for chunk in chunks])

        q_interp = np.vstack(list(results))

        if updateSelf:
            self.t_interp = t_interp
            self.ecef_interp = ecef_interp
            self.lla_interp = lla_interp
            self.q_interp = q_interp
        else:
            return t_interp, ecef_interp, q_interp

    def estimate_q_l2e(self, interp=True):
        """
        Estimate quaternion for rotation from local NED to ECEF.
        Parallelized across CPU cores.
        """

        if interp:
            n = len(self.t_interp)
        else:
            n = len(self.time)

        num_workers = multiprocessing.cpu_count() - 1 
        chunk_size = int(np.ceil(n / num_workers))
        if interp:
            chunks = [self.lla_interp[i:i+chunk_size] for i in range(0, n, chunk_size)]
        else:
            chunks = [self.lla[i:i+chunk_size] for i in range(0, n, chunk_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = ex.map(_compute_q_l2e_chunk, chunks)

        self.q_l2e = np.vstack(list(results))

def compareTrajectories(refTrj, comparedTrj, cfg):
    """
    Compare each trajectory in comparedTrj with a reference trajectory refTrj.
    Generates error plots and summary statistics tables.
    """
    outpath = cfg['outpath'] if 'outpath' in cfg else None
    name = cfg['name'] if 'name' in cfg else ''

    if refTrj.ecef_interp is None:
        raise ValueError("Reference trajectory must be interpolated before comparison.")
    for trj in comparedTrj:
        if trj.ecef_interp is None:
            raise ValueError("Compared trajectory must be interpolated before comparison.")

    pos_error_norms, rpy_errors = [], []

    for trj in comparedTrj:
        err_ecef = refTrj.ecef_interp - trj.ecef_interp
        pos_error_norms.append(np.linalg.norm(err_ecef, axis=1))
        rpy_errors.append(compute_rpyError(refTrj.q_interp, trj.q_interp))

    _plot_error_traces(refTrj.t_interp, comparedTrj, rpy_errors, refTrj, outpath, name)
    _plot_histograms(pos_error_norms, rpy_errors, outpath, name)
    _plot_statistics_tables(pos_error_norms, rpy_errors, comparedTrj, outpath, name)

def _plot_error_traces(t, comparedTrj, rpy_errors, refTrj, outpath=None, name=None):
    fig, ax = plt.subplots(3, 3, figsize=(16, 10))
    labels = ['x1 [m]', 'x2 [m]', 'x3 [m]']
    ned_labels = ['North [m]', 'East [m]', 'Down [m]']
    rpy_labels = ['Roll [$\\circ$]', 'Pitch [$\\circ$]', 'Yaw [$\\circ$]']

    for i, trj in enumerate(comparedTrj):
        err_ecef = refTrj.ecef_interp - trj.ecef_interp
        err_local_ned = np.array([quat2dcm(refTrj.q_l2e[j]) @ err_ecef[j] for j in range(len(err_ecef))])
        rpy_bias = np.median(rpy_errors[i], axis=0)

        for j in range(3):
            ax[j, 0].plot(t, err_ecef[:, j], label=f"Trj {i+1}")
            ax[j, 1].plot(t, err_local_ned[:, j], label=f"Trj {i+1}")
            ax[j, 2].plot(t, rpy_errors[i][:, j], label=f"Trj {i+1}")
            ax[j, 2].hlines(rpy_bias[j], t[0], t[-1], colors='r', linestyles='dashed', label=f"Trj {i+1} rpy bias: {rpy_bias} [deg]")

    for j in range(3):
        ax[j, 0].set_ylabel(labels[j])
        ax[j, 1].set_ylabel(ned_labels[j])
        ax[j, 2].set_ylabel(rpy_labels[j])
    ax[0, 0].set_title('ECEF errors w.r.t. Reference')
    ax[0, 1].set_title('local NED errors w.r.t. Reference')
    ax[0, 2].set_title('RPY errors w.r.t. Reference')
    ax[2, 0].set_xlabel('Time [s]')
    ax[2, 1].set_xlabel('Time [s]')
    ax[2, 2].set_xlabel('Time [s]')
    ax[0, 2].legend()

    plt.subplots_adjust(wspace=0.25)

    if outpath is not None:
        plt.savefig(f"{outpath}/{name}_error_traces.svg", dpi=300)

    

def _plot_histograms(pos_error_norms, rpy_errors, outpath=None, name=None):
    fig, ax = plt.subplots(4, 1, figsize=(8, 10))
    rpy_all = np.vstack(rpy_errors)
    pos_all = np.concatenate(pos_error_norms)
    rpy_bins = np.linspace(np.min(rpy_all), np.max(rpy_all), 50)
    pos_bins = np.linspace(np.min(pos_all), np.max(pos_all), 50)

    for i in range(len(rpy_errors)):
        ax[0].hist(rpy_errors[i][:, 0], bins=rpy_bins, alpha=0.5, label=f"Trj {i+1}")
        ax[1].hist(rpy_errors[i][:, 1], bins=rpy_bins, alpha=0.5, label=f"Trj {i+1}")
        ax[2].hist(rpy_errors[i][:, 2], bins=rpy_bins, alpha=0.5, label=f"Trj {i+1}")
        ax[3].hist(pos_error_norms[i], bins=pos_bins, alpha=0.5, label=f"Trj {i+1}")

    xlabel = ['Roll Error [$\\circ$]', 'Pitch Error [$\\circ$]', 'Yaw Error [$\\circ$]', 'Position Error Norm [m]']
    for i in range(4):
        ax[i].set_xlabel(xlabel[i])
        ax[i].set_ylabel('Frequency')
    ax[0].legend()
    plt.subplots_adjust(hspace=0.35)
    if outpath is not None:
        plt.savefig(f"{outpath}/{name}_error_histograms.svg", dpi=300)

def _plot_statistics_tables(pos_error_norms, rpy_errors, comparedTrj, outpath=None, name=None):
    n = len(comparedTrj)
    metrics = [np.zeros((n, 4)) for _ in range(7)]  # rmse, q5, q25, q50, q75, q95, std
    rmse, q5, q25, q50, q75, q95, std = metrics

    for i in range(n):
        abs_rpy = np.abs(rpy_errors[i])
        rmse[i, 0] = np.sqrt(np.mean((pos_error_norms[i])**2))
        std[i, 0] = np.std(pos_error_norms[i])
        q5[i, 0] = np.quantile(pos_error_norms[i], 0.05)
        q25[i, 0] = np.quantile(pos_error_norms[i], 0.25)
        q50[i, 0] = np.quantile(pos_error_norms[i], 0.5)
        q75[i, 0] = np.quantile(pos_error_norms[i], 0.75)
        q95[i, 0] = np.quantile(pos_error_norms[i], 0.95)

        for j in range(3):
            rmse[i, j+1] = np.sqrt(np.mean((abs_rpy[:, j])**2))
            std[i, j+1] = np.std(abs_rpy[:, j])
            q5[i, j+1] = np.quantile(abs_rpy[:, j], 0.05)
            q25[i, j+1] = np.quantile(abs_rpy[:, j], 0.25)
            q50[i, j+1] = np.quantile(abs_rpy[:, j], 0.5)
            q75[i, j+1] = np.quantile(abs_rpy[:, j], 0.75)
            q95[i, j+1] = np.quantile(abs_rpy[:, j], 0.95)

    stats = [rmse, q5, q25, q50, q75, q95, std]
    stats = [np.round(m, 4) for m in stats]

    col_labels = ["RMSE", "Q5", "Q25", "Q50", "Q75", "Q95", "STD"]
    row_labels = [f"Trj {i+1}" for i in range(n)]

    categories = ["Position Error Norm [m]", "Roll Error [deg]", "Pitch Error [deg]", "Yaw Error [deg]"]
    fig, axs = plt.subplots(2, 2, figsize=(16, 6))

    for ax, j, title in zip(axs.flat, range(4), categories):
        data = np.column_stack([s[:, j] for s in stats])
        ax.axis("off")
        ax.set_title(title, fontweight='bold')
        table = ax.table(cellText=data, colLabels=col_labels, rowLabels=row_labels, loc='center', cellLoc='center')
        table.scale(1, 2)
    if outpath is not None:
        plt.savefig(f"{outpath}/{name}_error_statistics.svg", dpi=300)
    plt.tight_layout()
    plt.show()

def compute_rpyError(q1, q2):
    """
    Compute the RPY error between two quaternions.
    :param q1: Nx4 quaternion array
    :param q2: Nx4 quaternion array
    :return: Nx3 RPY error in degrees
    """
    assert len(q1) == len(q2), "Quaternions must be of same length"
    rpy_error = np.zeros((len(q1), 3))
    for i in range(len(q1)):

        q_diff = quatMult(quatInv(q1[i]), q2[i])

        R = quat2dcm(q_diff)
        rpy_error[i] = [R[1, 2], R[2, 0], R[0, 1]]
    rpy_error *= 180 / np.pi
    if np.max(np.abs(rpy_error)) > 3:
        print("Warning, max RPY error > 3 degree, approximated angular error might be inaccurate")
    return rpy_error

def load_trajectory(entry):
    if entry['type'] == 'SBET':
        return Trajectory.fromSBET(entry['path'], entry['cfg'])
    elif entry['type'] == 'ASCII':
        return Trajectory.fromASCII(entry['path'], entry['cfg'])
    elif entry['type'] == 'DN':
        return Trajectory.fromDN(entry['path'], entry['cfg'])
    else:
        raise ValueError(f"Unsupported trajectory type: {entry['type']}")