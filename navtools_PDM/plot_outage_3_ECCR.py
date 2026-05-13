import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

T_OUTAGE_START = 315642
T_OUTAGE_END   = 316070
T0 = T_OUTAGE_START
def tr(t): return t - T0

SCANLINES = {
    1000: (315632.0, 315662.3),
    2000: (315662.3, 315717.2),
    3000: (315717.2, 315767.4),
    4000: (315767.4, 315832.1),
    5000: (315832.1, 315863.7),
    6000: (315863.7, 315944.6),
    7000: (315944.6, 315987.8),
    8000: (315987.8, 316080.0),
}
PAIRS = [
    (1000, 315632.0, 315662.3, 2000, 315662.3, 315717.2),
    (2000, 315662.3, 315717.2, 3000, 315717.2, 315767.4),
    (1000, 315644,  315653,   4000, 315796,   315808),
    (5000, 315845,  315860,   6000, 315915,   315927),
    (4000, 315812,  315830,   8000, 316051,   316066),
    (7000, 315972,  315978,   8000, 316024,   316032),
    (5000, 315832,  315850,   8000, 316035,   316049),
    (6000, 315922,  315936,   8000, 316035,   316049),
    (6000, 315935,  315943,   7000, 315951,   315956),
]

COLORS = [
    "#007480",  # Canard
    "#B51F1F",  # Groseille
    "#413D3A",  # Ardoise
    "#00A79F",  # Léman
    "#FF0000",  # Rouge
    "#CAC7C7",  # Perle
    "#007480",  # Canard (bis)
    "#B51F1F",  # Groseille (bis)
    "#00A79F",  # Léman (bis)
]

HATCHES = ["///", "\\\\\\", "...", None, None, None, None, None, None]

BRIDGE_OFFSETS = [0.55, 0.55, 0.55, 0.55, 0.4, 0.95, 0.75, 1.35, 0.55]

SCAN_IDS = sorted(SCANLINES.keys())
y_map = {sid: i for i, sid in enumerate(SCAN_IDS)}
N = len(SCAN_IDS)

def intervals_overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1

scan_pair_windows = defaultdict(list)
for idx, pair in enumerate(PAIRS):
    sA, tsA, teA, sB, tsB, teB = pair
    scan_pair_windows[sA].append((idx, tsA, teA, "A"))
    scan_pair_windows[sB].append((idx, tsB, teB, "B"))

segment_slot = {}
for sid, entries in scan_pair_windows.items():
    used = [False] * len(entries)
    groups = []
    for i in range(len(entries)):
        if used[i]: continue
        group = [i]
        used[i] = True
        for j in range(i+1, len(entries)):
            if used[j]: continue
            for k in group:
                i0, t0_, t1_, _ = entries[k]
                j0, t0j, t1j, _ = entries[j]
                if intervals_overlap(t0_, t1_, t0j, t1j):
                    group.append(j)
                    used[j] = True
                    break
        groups.append(group)

    for group in groups:
        n = len(group)
        for slot, g_idx in enumerate(group):
            pidx, ts, te, role = entries[g_idx]
            segment_slot[(pidx, role)] = (slot, n)

def draw_segment(ax, y, ts, te, color, hatch, slot, n_slots):
    H_FULL = 0.4
    h      = H_FULL / n_slots
    y0     = y + H_FULL/2 - h/2 - slot * h
    w      = tr(te) - tr(ts)
    left   = tr(ts)

    if hatch:
        ax.barh(y0, w, left=left, height=h,
                color="white", edgecolor=color, lw=1.2, zorder=3)
        ax.barh(y0, w, left=left, height=h,
                color="none", edgecolor=color, lw=0.8, hatch=hatch, zorder=3)
    else:
        ax.barh(y0, w, left=left, height=h,
                color=color, alpha=0.88, edgecolor="white", lw=0.5, zorder=3)

fig, ax = plt.subplots(figsize=(13, 5.8))

ax.axvspan(tr(T_OUTAGE_START), tr(T_OUTAGE_END), color="#f5f6fa", zorder=0)
for x in [tr(T_OUTAGE_START), tr(T_OUTAGE_END)]:
    ax.axvline(x, color="#b2bec3", lw=0.9, ls="--", zorder=1)

for sid, (ts, te) in SCANLINES.items():
    y = y_map[sid]
    ax.barh(y, tr(te)-tr(ts), left=tr(ts), height=0.4,
            color="#dfe6e9", edgecolor="#95a5a6", lw=0.8, zorder=2)

for idx, (pair, color, hatch) in enumerate(zip(PAIRS, COLORS, HATCHES)):
    sA, tsA, teA, sB, tsB, teB = pair
    yA = y_map[sA]
    yB = y_map[sB]

    slot_A, n_A = segment_slot.get((idx, "A"), (0, 1))
    slot_B, n_B = segment_slot.get((idx, "B"), (0, 1))

    draw_segment(ax, yA, tsA, teA, color, hatch, slot_A, n_A)
    draw_segment(ax, yB, tsB, teB, color, hatch, slot_B, n_B)

for idx, (pair, color, hatch) in enumerate(zip(PAIRS, COLORS, HATCHES)):
    sA, tsA, teA, sB, tsB, teB = pair
    yA = y_map[sA]
    yB = y_map[sB]

    xA       = tr((tsA + teA) / 2)
    xB       = tr((tsB + teB) / 2)
    x_mid    = (xA + xB) / 2
    y_bridge = max(yA, yB) + BRIDGE_OFFSETS[idx]
    lw_b     = 1.0 if hatch else 1.2

    ax.plot([xA, xA], [yA+0.2, y_bridge], color=color, lw=lw_b, zorder=4)
    ax.plot([xB, xB], [yB+0.2, y_bridge], color=color, lw=lw_b, zorder=4)
    ax.plot([xA, xB], [y_bridge, y_bridge],
            color=color, lw=lw_b, zorder=4,
            ls="--" if hatch else "-")
    ax.text(x_mid, y_bridge + 0.06, f"({idx+1})",
            ha="center", va="bottom", fontsize=7.5,
            color=color, fontweight="bold")

ax.set_yticks(list(y_map.values()))
ax.set_yticklabels([f"Scan {sid}" for sid in SCAN_IDS], fontsize=9)

legend_handles = []
for i, (pair, color, hatch) in enumerate(zip(PAIRS, COLORS, HATCHES)):
    if hatch:
        patch = mpatches.Patch(facecolor="white", edgecolor=color,
                               hatch=hatch, label=f"({i+1})  Scan {pair[0]} ↔ {pair[3]}")
    else:
        patch = mpatches.Patch(facecolor=color, alpha=0.88,
                               label=f"({i+1})  Scan {pair[0]} ↔ {pair[3]}")
    legend_handles.append(patch)

ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
          framealpha=0.95, ncol=2,
          title="Matched pairs", title_fontsize=8.5)

ax.set_xlabel("Time relative to outage start (s)", fontsize=9)
ax.set_xlim(tr(315618), tr(316090))
ax.set_ylim(-0.55, N + 0.55)
ax.grid(axis="x", color="#dfe6e9", lw=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax.text(tr(T_OUTAGE_START), N + 0.28, "Outage start", fontsize=11,
        color="#7f8c8d", va="bottom", ha="center", fontstyle="italic")
ax.text(tr(T_OUTAGE_END),   N + 0.28, "Outage end",   fontsize=11,
        color="#7f8c8d", va="bottom", ha="center", fontstyle="italic")

plt.tight_layout()
plt.savefig("/home/b085164/PDM_Romain_Defferrard/images/crossing_3_ECCR_final.png", bbox_inches="tight", dpi=400)
print("Done")