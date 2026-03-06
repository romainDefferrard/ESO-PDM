from __future__ import annotations

from pathlib import Path
import numpy as np

def _is_probably_lla(lat: np.ndarray, lon: np.ndarray) -> bool:
    # latitude ~ [-90, 90], longitude ~ [-180, 180]
    return (
        np.all(np.isfinite(lat)) and np.all(np.isfinite(lon)) and
        (np.nanmin(lat) > -90) and (np.nanmax(lat) < 90) and
        (np.nanmin(lon) > -180) and (np.nanmax(lon) < 180)
    )

def _enu_basis(lat0_deg: float, lon0_deg: float):
    """
    Returns 3 unit vectors (ECEF) for ENU basis at (lat0, lon0).
    """
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    sL, cL = np.sin(lat0), np.cos(lat0)
    sλ, cλ = np.sin(lon0), np.cos(lon0)

    e = np.array([-sλ, cλ, 0.0])
    n = np.array([-sL*cλ, -sL*sλ, cL])
    u = np.array([cL*cλ, cL*sλ, sL])
    return e, n, u

def _lla_to_ecef(lat_deg, lon_deg, h_m):
    # WGS84
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    s = np.sin(lat)
    c = np.cos(lat)

    N = a / np.sqrt(1 - e2 * s * s)

    x = (N + h_m) * c * np.cos(lon)
    y = (N + h_m) * c * np.sin(lon)
    z = (N * (1 - e2) + h_m) * s
    return np.vstack([x, y, z]).T

def _ecef_to_lla(xyz):
    # WGS84, Bowring-like iterative
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b*b)/(a*a)
    ep2 = (a*a)/(b*b) - 1

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    lon = np.arctan2(y, x)
    p = np.sqrt(x*x + y*y)

    # initial lat
    theta = np.arctan2(z * a, p * b)
    sT, cT = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + ep2 * b * sT**3, p - e2 * a * cT**3)

    # height
    s = np.sin(lat)
    N = a / np.sqrt(1 - e2 * s*s)
    h = p / np.cos(lat) - N

    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    return lat_deg, lon_deg, h

def write_gps_cycle_slips(
    gps_in: str | Path,
    gps_out: str | Path,
    slips: list[dict],
    delimiter: str = ",",
):
    """
    Supports both:
      - projected meters: t, x, y, z  (applies dx/dy/dz directly)
      - LLA degrees:      t, lat, lon, h (applies dx/dy/dz in meters using local ENU)
    slips: dict keys:
      t0, duration (s or None), dx, dy, dz (meters), shape ("step"|"ramp"), tau (s for ramp)
    """
    gps_in = Path(gps_in)
    gps_out = Path(gps_out)
    gps_out.parent.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(str(gps_in), delimiter=delimiter)
    if data.ndim == 1:
        data = data[None, :]

    t = data[:, 0]
    c1 = data[:, 1].copy()
    c2 = data[:, 2].copy()
    c3 = data[:, 3].copy()

    is_lla = _is_probably_lla(c1, c2)

    if is_lla:
        # Interpret as: lat, lon, h
        lat = c1
        lon = c2
        h = c3

        # Use first sample as local tangent origin
        lat0, lon0, h0 = float(lat[0]), float(lon[0]), float(h[0])

        xyz = _lla_to_ecef(lat, lon, h)
        xyz0 = _lla_to_ecef(np.array([lat0]), np.array([lon0]), np.array([h0]))[0]

        e, n, u = _enu_basis(lat0, lon0)
        # Convert ECEF deltas -> ENU
        dxyz = xyz - xyz0
        east  = dxyz @ e
        north = dxyz @ n
        up    = dxyz @ u

        # Apply slips in ENU meters
        E = east.copy()
        N = north.copy()
        U = up.copy()

        for s in slips:
            t0 = float(s["t0"])
            duration = s.get("duration", None)
            dx = float(s.get("dx", 0.0))  # interpreted as East meters
            dy = float(s.get("dy", 0.0))  # North meters
            dz = float(s.get("dz", 0.0))  # Up meters
            shape = s.get("shape", "step")
            tau = float(s.get("tau", 10.0))

            if duration is None:
                mask = t >= t0
                tt = t[mask] - t0
            else:
                t1 = t0 + float(duration)
                mask = (t >= t0) & (t <= t1)
                tt = t[mask] - t0

            if not np.any(mask):
                continue

            if shape == "step":
                w = np.ones_like(tt)
            elif shape == "ramp":
                w = np.exp(-tt / max(tau, 1e-6))
            else:
                raise ValueError(f"Unknown shape={shape}")

            E[mask] += dx * w
            N[mask] += dy * w
            U[mask] += dz * w

        # ENU -> ECEF -> LLA
        xyz_new = xyz0 + np.outer(E, e) + np.outer(N, n) + np.outer(U, u)
        lat_new, lon_new, h_new = _ecef_to_lla(xyz_new)

        out = data.copy()
        out[:, 1] = lat_new
        out[:, 2] = lon_new
        out[:, 3] = h_new

    else:
        # Interpret as meters: x, y, z
        xyz = np.vstack([c1, c2, c3]).T

        for s in slips:
            t0 = float(s["t0"])
            duration = s.get("duration", None)
            dx = float(s.get("dx", 0.0))
            dy = float(s.get("dy", 0.0))
            dz = float(s.get("dz", 0.0))
            shape = s.get("shape", "step")
            tau = float(s.get("tau", 10.0))

            if duration is None:
                mask = t >= t0
                tt = t[mask] - t0
            else:
                t1 = t0 + float(duration)
                mask = (t >= t0) & (t <= t1)
                tt = t[mask] - t0

            if not np.any(mask):
                continue

            if shape == "step":
                w = np.ones_like(tt)
            elif shape == "ramp":
                w = np.exp(-tt / max(tau, 1e-6))
            else:
                raise ValueError(f"Unknown shape={shape}")

            xyz[mask, 0] += dx * w
            xyz[mask, 1] += dy * w
            xyz[mask, 2] += dz * w

        out = data.copy()
        out[:, 1:4] = xyz

    np.savetxt(str(gps_out), out, delimiter=delimiter, fmt="%.6f")
    return gps_out