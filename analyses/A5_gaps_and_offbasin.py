"""Describe the longest observation gaps and extraction-box-only retrieval days.

The checks compare extraction-box availability with any valid subbasin value.
They also count overlap with dates omitted by the reduced schedules. These are
spatial-support and calendar diagnostics, rather than independent accuracy tests.

Author: Sebastian R.O. Marshall."""
import pandas as pd, numpy as np, glob
from pathlib import Path
T = Path(__file__).resolve().parents[1]
for r in ("9km", "1km", "400m"):
    f = pd.read_csv(T / f"data/enkf/inputs_{r}.csv", parse_dates=["date"]); m = f.obs_available.astype(bool).to_numpy()
    idx = np.flatnonzero(m); gaps = np.diff(idx); k = gaps.argmax()
    print(f"{r}: longest gap {gaps[k]} days, from {f.date[idx[k]].date()} (last obs) to {f.date[idx[k+1]].date()} (next obs)")
sub = {}
for r in ("9km", "1km", "400m"):
    fs = sorted(glob.glob(str(T / f"data/smap_subbasin_daily/*{r}*")))
    if not fs: print(r, "no subbasin file"); continue
    s = pd.read_csv(fs[0]); dcol = [c for c in s.columns if "date" in c.lower()][0]; s[dcol] = pd.to_datetime(s[dcol])
    v = s.set_index(dcol).drop(columns=[c for c in s.columns if c != dcol and not np.issubdtype(s[c].dtype, np.number)], errors="ignore")
    sub[r] = v.notna().any(axis=1)
    f = pd.read_csv(T / f"data/enkf/inputs_{r}.csv", parse_dates=["date"]).set_index("date"); m = f.obs_available.astype(bool)
    anyv = sub[r].reindex(m.index).fillna(False).astype(bool)
    off = m & ~anyv; print(f"{r}: assimilated days {int(m.sum())}, subbasin-valid days {int(anyv.sum())}, box-only days {int(off.sum())}, file {Path(fs[0]).name}")
f9 = pd.read_csv(T / "data/enkf/inputs_9km.csv", parse_dates=["date"]).set_index("date"); f1 = pd.read_csv(T / "data/enkf/inputs_1km.csv", parse_dates=["date"]).set_index("date"); f4 = pd.read_csv(T / "data/enkf/inputs_400m.csv", parse_dates=["date"]).set_index("date")
m9, m1, m4 = (x.obs_available.astype(bool) for x in (f9, f1, f4))
if "9km" in sub:
    off9 = m9 & ~sub["9km"].reindex(m9.index).fillna(False).astype(bool)
    drop1 = m9 & ~m1; drop4 = m9 & ~m4
    print(f"9 km days dropped by the 1 km calendar: {int(drop1.sum())}, of which box-only (no subbasin value): {int((drop1 & off9).sum())}")
    print(f"9 km days dropped by the 400 m calendar: {int(drop4.sum())}, of which box-only: {int((drop4 & off9).sum())}")
