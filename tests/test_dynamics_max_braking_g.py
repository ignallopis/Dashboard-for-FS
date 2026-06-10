from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.driver as drv
from utils import cols_to_numpy, ensure_complete_laps_df, load_data, unique_laps


class MaxBrakingGPerLapFigureTest(unittest.TestCase):
    def test_uses_per_lap_braking_p5_and_raising_average(self) -> None:
        run_name = "Cerpa"
        df = ensure_complete_laps_df(load_data("data/Cerpa_FSG.csv", complete_laps_only=False))

        fig, kpis = drv.max_braking_g_per_lap_fig({run_name: df})

        arr = cols_to_numpy(df, ["laps", "Filtering_VN_ax", "Brake"])
        lap_ids = unique_laps(arr["laps"]).astype(int)
        expected_laps: list[int] = []
        expected_min_g: list[float] = []
        for lap_id in lap_ids:
            mask = (
                (arr["laps"] == lap_id)
                & np.isfinite(arr["Filtering_VN_ax"])
                & (arr["Brake"] > 5.0)
                & (arr["Filtering_VN_ax"] < -1.0)
            )
            if int(mask.sum()) < 5:
                continue
            expected_laps.append(int(lap_id))
            expected_min_g.append(
                float(np.nanpercentile(arr["Filtering_VN_ax"][mask], 5.0) / drv.G_MPS2)
            )

        expected_raise = (
            np.cumsum(expected_min_g) / np.arange(1, len(expected_min_g) + 1)
        ).tolist()

        self.assertEqual(len(fig.data), 2)
        raw_trace = fig.data[0]
        raising_trace = fig.data[1]

        self.assertEqual(list(raw_trace.x), expected_laps)
        self.assertTrue(np.allclose(raw_trace.y, expected_min_g, atol=1e-9))
        self.assertEqual(list(raising_trace.x), expected_laps)
        self.assertTrue(np.allclose(raising_trace.y, expected_raise, atol=1e-9))

        run_kpis = kpis["runs"][run_name]
        self.assertEqual(run_kpis["valid_laps"], len(expected_laps))
        self.assertAlmostEqual(run_kpis["best_max_braking_g"], min(expected_min_g), places=9)
        self.assertAlmostEqual(run_kpis["raising_avg_last_g"], expected_raise[-1], places=9)


if __name__ == "__main__":
    unittest.main()
