"""Parser for the SWAT2012 reach output file, ``output.rch``.

``output.rch`` is the only model output the streamflow analysis needs, and at the Mekong
geometry it is several gigabytes: 540 reaches written every simulated day for 24 reported
years. The parser is therefore a single streaming pass that keeps only the requested
reaches.

Format notes that matter:

* Nine header lines precede the data, but every data line begins with the literal
  ``REACH``, so filtering on that prefix is safer than counting header lines and is what
  the published run did.
* Fields are ``RCH GIS MON AREAkm2 FLOW_INcms FLOW_OUTcms``, giving seven whitespace
  tokens per line once the ``REACH`` label is counted. Discharge is token index 6.
  Index 5 is *inflow*, and reading it instead is a silent error: the series still looks
  like a hydrograph.
* ``MON`` restarts every year and is a month or day-of-year depending on the print code,
  so it cannot be used to count days. Days are counted by watching the reach index wrap.
* Output begins after the warm-up years skipped by ``NYSKIP`` in ``file.cio``, not at the
  simulation start year.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd

from .config import MEKONG_2024, STATIONS, SwatRunSpec

__all__ = ["parse_output_rch", "station_frame"]

#: Whitespace token index of FLOW_OUTcms on a ``REACH`` line.
FLOW_OUT_INDEX = 6

#: Whitespace token index of FLOW_INcms, kept named so it is never confused with the above.
FLOW_IN_INDEX = 5


def parse_output_rch(
    path: Union[str, Path],
    reaches: Iterable[int],
    spec: SwatRunSpec = MEKONG_2024,
    column: int = FLOW_OUT_INDEX,
) -> pd.DataFrame:
    """Extract daily discharge for selected reaches.

    Parameters
    ----------
    path:
        Path to ``output.rch``.
    reaches:
        Reach identifiers to keep. Everything else is discarded while streaming.
    spec:
        Model calendar, used to date the first reported day.
    column:
        Token index to read. Defaults to :data:`FLOW_OUT_INDEX`; pass
        :data:`FLOW_IN_INDEX` only if inflow is genuinely wanted.

    Returns
    -------
    pandas.DataFrame
        Indexed by date, one column per requested reach named ``reach_<id>``, in m3/s.
        Reaches absent from the file appear as all-NaN columns rather than vanishing, so
        that a missing gauge is visible instead of silently dropped.

    Raises
    ------
    ValueError
        If the file contains no ``REACH`` records at all.
    """
    path = Path(path)
    wanted = list(dict.fromkeys(int(r) for r in reaches))
    wanted_set = set(wanted)
    series: Dict[int, list] = {r: [] for r in wanted}

    day = 0
    prev_reach = 10**9
    n_records = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("REACH"):
                continue
            parts = line.split()
            if len(parts) <= column:
                continue
            try:
                reach = int(parts[1])
                value = float(parts[column])
            except (ValueError, IndexError):
                continue
            n_records += 1
            if reach <= prev_reach:      # the reach index wrapped, so a new day started
                day += 1
            prev_reach = reach
            if reach in wanted_set:
                series[reach].append((day, value))

    if n_records == 0:
        raise ValueError(
            f"{path} contains no REACH records. Either the run did not complete or the "
            "print code in file.cio suppressed reach output."
        )

    n_days = day
    dates = pd.date_range(f"{spec.output_start_year}-01-01", periods=n_days, freq="D")
    out = pd.DataFrame(index=dates)
    out.index.name = "date"
    for reach in wanted:
        values = np.full(n_days, np.nan)
        for d, v in series[reach]:
            if 1 <= d <= n_days:
                values[d - 1] = v
        out[f"reach_{reach}"] = values
    return out


def station_frame(
    path: Union[str, Path],
    spec: SwatRunSpec = MEKONG_2024,
) -> pd.DataFrame:
    """Parse ``output.rch`` straight into a frame keyed by gauge name.

    Convenience wrapper over :func:`parse_output_rch` using the five Mekong River
    Commission gauges declared in :data:`mekong_da.config.STATIONS`.
    """
    frame = parse_output_rch(path, [s.reach for s in STATIONS], spec=spec)
    return frame.rename(columns={f"reach_{s.reach}": s.key for s in STATIONS})
