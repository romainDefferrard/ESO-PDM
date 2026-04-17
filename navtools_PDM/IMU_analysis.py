"""
imu_boresight_analysis.py
=========================
Analyse de l'alignement APX15 → AIRINS body frame.

4 scénarios : sens du boresight APX/PUCK × avec/sans boresight PUCK/AIRINS
+ correction bras de levier
+ cross-corrélation sur les 3 accéléromètres
+ fenêtre courte configurable
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PATH_AIRINS      = '/media/b085164/LaCie/2026spring_RD/ECCR/ODyN/base/IMU.txt'
PATH_APX         = '/media/b085164/LaCie/2026spring_RD/ECCR/RAW/03_IMU/APX/imu_ECCR_MLS.txt'
OUT_DIR          = Path('/media/b085164/LaCie/2026spring_RD/ECCR/ODyN/APX/outputs')
WINDOW_DURATION  = 40.0   # secondes
MAX_LAG_S        = 0.5    # plage cross-corrélation (s)

# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS ROTATION — convention lidar.py
# ══════════════════════════════════════════════════════════════════════════════
def R1(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,sa],[0,-sa,ca]])

def R2(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,0,-sa],[0,1,0],[sa,0,ca]])

def R3(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])

def make_boresight(roll_rad, pitch_rad, yaw_rad):
    """Convention lidar.py : (R1@R2@R3).T"""
    return (R1(roll_rad) @ R2(pitch_rad) @ R3(yaw_rad)).T

# ══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES DE MONTAGE
# ══════════════════════════════════════════════════════════════════════════════
R_mount_PUCK = np.array([
    [ 0.97236992,  0.0,  0.233445364],
    [ 0.0,        -1.0,  0.0        ],
    [ 0.233445364, 0.0, -0.97236992 ]
], dtype=float)

# Boresight PUCK/AIRINS — scanner_PUCK.yml (radians)
R_bs_puck_airins = make_boresight(0.00555947, 0.00293948, -0.00591964)

# Boresight APX15/PUCK
R_bs_apx = make_boresight(
    np.radians(-0.1635),
    np.radians(-0.11584),
    np.radians(-0.14407)
)

# Changement de convention APX axes → frame platine PUCK
# (validé empiriquement : R_swap = Rz+90°, R_flip = diag(-1,1,-1))
R_flip = np.diag([-1., 1., -1.])
R_swap = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])

# Géométrie nominale APX → body AIRINS
R_mount_apx = R_swap @ R_mount_PUCK @ R_flip

# Bras de levier AIRINS → APX (NED body AIRINS, mètres)
LEVER_ARM = np.array([0.206, -0.014, -0.743])

# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DES 4 SCÉNARIOS
# ══════════════════════════════════════════════════════════════════════════════
scenarios = {}

for bs_label, R_bs in [('apx2puck', R_bs_apx), ('puck2apx', R_bs_apx.T)]:
    for mount_label in ['sans_bs_puck', 'avec_bs_puck']:
        if mount_label == 'sans_bs_puck':
            R_total = R_bs @ R_mount_apx
        else:
            R_mount_apx_corr = R_swap @ (R_bs_puck_airins @ R_mount_PUCK) @ R_flip
            R_total = R_bs @ R_mount_apx_corr

        name = f"{bs_label}__{mount_label}"
        det  = np.linalg.det(R_total)
        err  = np.max(np.abs(R_total @ R_total.T - np.eye(3)))
        scenarios[name] = dict(R_total=R_total, det=det, orth_err=err)
        print(f"  {name:40s}  det={det:+.6f}  err={err:.1e}")

# ── Scénario 5 : correction manuelle du biais résiduel odyn ──────────────────
# Valeurs lues sur le plot odyn (roll=-0.40°, pitch=+8.30°, yaw=-0.60°)
# On applique la correction inverse pour annuler ces biais
pitch_fix = np.radians(-8.30)
R_pitch_fix = R2(pitch_fix)   # rotation pure autour de Y

# Injecter dans R_mount_PUCK
R_mount_PUCK_fixed = R_pitch_fix @ R_mount_PUCK

# Reconstruire R_mount_apx avec le mount corrigé
R_mount_apx_fixed = R_swap @ R_mount_PUCK_fixed @ R_flip

# R_total avec correction pitch dans le mount
R_total_manual = R_bs_apx @ R_mount_apx_fixed

name = 'manual_rpy_correction'
det  = np.linalg.det(R_total_manual)
err  = np.max(np.abs(R_total_manual @ R_total_manual.T - np.eye(3)))
scenarios[name] = dict(R_total=R_total_manual, det=det, orth_err=err)
print(f"  {name:40s}  det={det:+.6f}  err={err:.1e}")

print()

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════
COLS = ['time','gyr_x','gyr_y','gyr_z','acc_x','acc_y','acc_z']
SIG  = COLS[1:]

print('Chargement...')
airins  = pd.read_csv(PATH_AIRINS, header=None, names=COLS, dtype=np.float64)
apx_raw = pd.read_csv(PATH_APX,    header=None, names=COLS, dtype=np.float64)
t_start = max(airins.time.iloc[0],  apx_raw.time.iloc[0])
t_end   = min(airins.time.iloc[-1], apx_raw.time.iloc[-1])
print(f'  Plage commune : {(t_end-t_start)/60:.1f} min\n')

# ══════════════════════════════════════════════════════════════════════════════
# FENÊTRE LA PLUS DYNAMIQUE
# ══════════════════════════════════════════════════════════════════════════════
air_c = airins[(airins.time >= t_start) & (airins.time <= t_end)].reset_index(drop=True)
air_c['energy'] = air_c['gyr_z']**2 * 0.5 + air_c['acc_y']**2 * 0.5
best_e, best_t0 = 0.0, t_start
for t0 in np.arange(t_start, t_end - WINDOW_DURATION, 2.0):
    e = air_c[(air_c.time >= t0) & (air_c.time < t0 + WINDOW_DURATION)]['energy'].mean()
    if e > best_e:
        best_e, best_t0 = e, t0

t_win_s = best_t0
t_win_e = best_t0 + WINDOW_DURATION
print(f'Fenêtre : [{t_win_s:.1f}, {t_win_e:.1f}] s  énergie={best_e:.5f}\n')

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION BRAS DE LEVIER
# ══════════════════════════════════════════════════════════════════════════════
def lever_arm_correction(df, R_mat, lever_body):
    r_apx = R_mat.T @ lever_body
    gyr = df[['gyr_x','gyr_y','gyr_z']].values
    acc = df[['acc_x','acc_y','acc_z']].values
    centrifugal = np.cross(gyr, np.cross(gyr, r_apx[np.newaxis,:]))
    out = df.copy()
    out[['acc_x','acc_y','acc_z']] = acc - centrifugal
    return out

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION + INTERPOLATION
# ══════════════════════════════════════════════════════════════════════════════
def transform_interp(apx_df, R_mat, t_grid, use_lever=False):
    df = apx_df.copy()
    if use_lever:
        df = lever_arm_correction(df, R_mat, LEVER_ARM)
    df[['gyr_x','gyr_y','gyr_z']] = (R_mat @ df[['gyr_x','gyr_y','gyr_z']].values.T).T
    df[['acc_x','acc_y','acc_z']] = (R_mat @ df[['acc_x','acc_y','acc_z']].values.T).T
    out = {'time': t_grid}
    for col in SIG:
        f = interp1d(df.time.values, df[col].values,
                     kind='linear', bounds_error=False, fill_value=np.nan)
        out[col] = f(t_grid)
    return pd.DataFrame(out)

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CORRÉLATION
# ══════════════════════════════════════════════════════════════════════════════
def xcorr_lag(ref, sig, dt, max_lag=MAX_LAG_S):
    mask = ~(np.isnan(ref) | np.isnan(sig))
    if mask.sum() < 50:
        return 0.0, 0.0
    a = ref[mask]; b = sig[mask]
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    cc   = correlate(a, b, mode='full')
    lags = correlation_lags(len(a), len(b), mode='full') * dt
    m    = np.abs(lags) <= max_lag
    best = np.argmax(np.abs(cc[m]))
    return float(lags[m][best]), float(cc[m][best] / len(a))

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════
def metrics(air, apx):
    out = {}
    for col in SIG:
        a, b = air[col].values, apx[col].values
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 10:
            out[col] = dict(corr=np.nan, rms=np.nan, bias=np.nan)
            continue
        c, _ = pearsonr(a[mask], b[mask])
        out[col] = dict(
            corr=c,
            rms =np.sqrt(np.mean((b[mask]-a[mask])**2)),
            bias=np.mean(b[mask]-a[mask])
        )
    return out

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
MARGIN = 2.0
air_win = airins[(airins.time >= t_win_s) & (airins.time <= t_win_e)].reset_index(drop=True)
apx_win = apx_raw[(apx_raw.time >= t_win_s-MARGIN) & (apx_raw.time <= t_win_e+MARGIN)].reset_index(drop=True)
t_grid  = air_win.time.values
dt      = float(np.median(np.diff(t_grid)))
t_rel   = t_grid - t_grid[0]

print(f"{'Scénario':42s}  {'lag ms':>7s}  "
      + "  ".join(f"{c:>6s}" for c in SIG))
print('-' * 100)

results = {}
for name, sc in scenarios.items():
    Rm = sc['R_total']
    a_no  = transform_interp(apx_win, Rm, t_grid, use_lever=False)
    a_lv  = transform_interp(apx_win, Rm, t_grid, use_lever=True)
    # Cross-corrélation sur acc_z (signal dominant = gravité)
    lag, _ = xcorr_lag(air_win['acc_z'].values, a_no['acc_z'].values, dt)
    m_no = metrics(air_win, a_no)
    m_lv = metrics(air_win, a_lv)
    results[name] = dict(apx_no=a_no, apx_lv=a_lv,
                         m_no=m_no, m_lv=m_lv,
                         lag_ms=lag*1000, R_total=Rm)
    corrs = [m_no[c]['corr'] for c in SIG]
    print(f"{name:42s}  {lag*1000:>+7.1f}  "
          + "  ".join(f"{c:>+6.3f}" for c in corrs))

print()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
C_AIR = '#2A6DB5'
C_APX = '#E07B54'
C_LV  = '#2E7D32'
C_ACC = ['#e41a1c', '#377eb8', '#4daf4a']   # acc_x, acc_y, acc_z

def plot_scenario(name, res, air_win, t_rel, path):
    a_no = res['apx_no']
    a_lv = res['apx_lv']
    m_no = res['m_no']
    m_lv = res['m_lv']
    lag  = res['lag_ms']

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'Scénario : {name}    (lag opt = {lag:+.1f} ms)',
                 fontsize=10, fontweight='bold')
    gs = gridspec.GridSpec(4, 3, hspace=0.48, wspace=0.32)

    cols_all = ['gyr_x','gyr_y','gyr_z','acc_x','acc_y','acc_z']
    units    = ['rad/s']*3 + ['m/s²']*3

    for i, (col, unit) in enumerate(zip(cols_all, units)):
        ax = fig.add_subplot(gs[i//3, i%3])
        ax.plot(t_rel, air_win[col].values, color=C_AIR, lw=0.9, alpha=0.95,
                label='AIRINS', zorder=3)
        ax.plot(t_rel, a_no[col].values, color=C_APX, lw=0.8, alpha=0.55,
                label='APX', zorder=1, ls='--')
        ax.plot(t_rel, a_lv[col].values, color=C_LV, lw=0.8, alpha=0.55,
                label='APX+LV', zorder=2, ls=':')
        cr  = m_no[col]['corr']
        cr2 = m_lv[col]['corr']
        ax.set_title(f'{col} [{unit}]\ncorr={cr:+.3f}  +LV={cr2:+.3f}', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.22)
        if i >= 3:
            ax.set_xlabel('t [s]', fontsize=8)

    # Cross-corrélation sur les 3 accéléromètres
    ax_cc = fig.add_subplot(gs[2, :])
    for acc_col, col_c in zip(['acc_x','acc_y','acc_z'], C_ACC):
        ref  = air_win[acc_col].values
        sig  = a_no[acc_col].values
        mask = ~(np.isnan(ref) | np.isnan(sig))
        if mask.sum() < 50:
            continue
        a_n = ref[mask]; b_n = sig[mask]
        a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-12)
        b_n = (b_n - b_n.mean()) / (b_n.std() + 1e-12)
        cc    = correlate(a_n, b_n, mode='full') / mask.sum()
        lgs   = correlation_lags(mask.sum(), mask.sum(), mode='full') * dt * 1000
        mask2 = np.abs(lgs) <= MAX_LAG_S * 1000
        ax_cc.plot(lgs[mask2], cc[mask2], lw=1.0, color=col_c, label=acc_col)
    ax_cc.axvline(lag, color='red', lw=1.2, ls='--', label=f'lag={lag:+.1f}ms')
    ax_cc.axvline(0,   color='k',   lw=0.6, ls=':')
    ax_cc.set_xlabel('Lag [ms]', fontsize=8)
    ax_cc.set_ylabel('Cross-corr. norm.', fontsize=8)
    ax_cc.set_title('Cross-corrélation accéléromètres (AIRINS ↔ APX)', fontsize=9)
    ax_cc.legend(fontsize=8, ncol=4)
    ax_cc.grid(True, alpha=0.22)

    # Résidus accéléromètres
    ax_r = fig.add_subplot(gs[3, :])
    for acc_col, col_c in zip(['acc_x','acc_y','acc_z'], C_ACC):
        ax_r.plot(t_rel, a_no[acc_col].values - air_win[acc_col].values,
                  lw=0.6, alpha=0.75, label=f'Δ{acc_col}', color=col_c)
    ax_r.axhline(0, color='k', lw=0.7, ls='--')
    ax_r.set_xlabel('t [s]', fontsize=8)
    ax_r.set_ylabel('Résidu [m/s²]', fontsize=8)
    ax_r.set_title('Résidus accéléromètres APX − AIRINS', fontsize=9)
    ax_r.legend(fontsize=8, ncol=3)
    ax_r.grid(True, alpha=0.22)

    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()

print('Génération des plots...')
OUT_DIR.mkdir(exist_ok=True)
for name, res in results.items():
    plot_scenario(name, res, air_win, t_rel, OUT_DIR / f'scenario_{name}.png')
    print(f'  scenario_{name}.png')

# ── Comparatif 3 accéléromètres × N scénarios ─────────────────────────────
n_sc   = len(results)
n_cols = min(4, n_sc)
n_rows = (n_sc + n_cols - 1) // n_cols

for acc_col, col_c in zip(['acc_x','acc_y','acc_z'], C_ACC):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows),
                              sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    fig.suptitle(f'Comparaison {n_sc} scénarios — {acc_col}\n'
                 f'Fenêtre {WINDOW_DURATION:.0f}s @ t₀={t_win_s:.0f}s',
                 fontsize=10, fontweight='bold')
    for idx, (name, res) in enumerate(results.items()):
        ax   = axes[idx]
        cr   = res['m_no'][acc_col]['corr']
        bias = res['m_no'][acc_col]['bias']
        lg   = res['lag_ms']
        ax.plot(t_rel, air_win[acc_col].values,
                color=C_AIR, lw=1.0, alpha=0.9, label='AIRINS', zorder=3)
        ax.plot(t_rel, res['apx_no'][acc_col].values,
                color=col_c, lw=0.8, alpha=0.65, label='APX', zorder=1, ls='--')
        ax.set_title(f'{name}\ncorr={cr:+.4f}  bias={bias:+.3f}  lag={lg:+.1f}ms',
                     fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.22)
        ax.set_ylabel(f'{acc_col} [m/s²]')
    for idx in range(n_sc, len(axes)):
        axes[idx].set_visible(False)
    for ax in axes[max(0, n_sc-n_cols):n_sc]:
        ax.set_xlabel('t [s]')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'comparaison_scenarios_{acc_col}.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  comparaison_scenarios_{acc_col}.png')

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*80)
print('RÉSUMÉ')
print('='*80)
print(f"\n{'Scénario':42s}  {'lag ms':>7s}  "
      + "  ".join(f"{c:>6s}" for c in SIG)
      + '  Σcorr')

best_name, best_score = '', -np.inf
for name, res in results.items():
    m    = res['m_no']
    cors = [m[c]['corr'] for c in SIG]
    score= sum(c for c in cors if not np.isnan(c))
    mark = ' ◄' if score > best_score else ''
    if score > best_score:
        best_score = score; best_name = name
    print(f"{name:42s}  {res['lag_ms']:>+7.1f}  "
          + "  ".join(f"{c:>+6.3f}" for c in cors)
          + f"  {score:>8.3f}{mark}")

print(f'\nMeilleur scénario : {best_name}')
R_best = results[best_name]['R_total']
print('R_total :')
for row in R_best:
    print('  ' + '  '.join(f'{v:+.8f}' for v in row))
print('='*80)

# ══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE DU FICHIER TRANSFORMÉ AVEC LE MEILLEUR SCÉNARIO
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nÉcriture du fichier transformé avec le scénario : {"manual_rpy_correction"}')

R_best = results["manual_rpy_correction"]['R_total']

apx_full = apx_raw.copy()
apx_full[['gyr_x','gyr_y','gyr_z']] = (R_best @ apx_raw[['gyr_x','gyr_y','gyr_z']].values.T).T
apx_full[['acc_x','acc_y','acc_z']] = (R_best @ apx_raw[['acc_x','acc_y','acc_z']].values.T).T

out_path = Path('/media/b085164/LaCie/2026spring_RD/ECCR/ODyN/APX/outputs') / 'IMU.txt'
apx_full.to_csv(out_path, header=False, index=False, float_format='%.10E')

check = pd.read_csv(out_path, header=None, names=COLS)
assert len(check) == len(apx_raw), 'ERREUR : nombre de lignes différent !'
print(f'  Fichier sauvegardé : {out_path}')
print(f'  Lignes : {len(apx_full):,}')
print(f'  R_total utilisée :')
for row in R_best:
    print('    ' + '  '.join(f'{v:+.8f}' for v in row))