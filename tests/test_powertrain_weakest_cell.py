import unittest

import numpy as np
import polars as pl

from src.powertrain import weakest_cell_fig


class WeakestCellFigureTest(unittest.TestCase):
    def test_discharge_floor_reference_is_3_75_v(self) -> None:
        samples = 240
        lap_ids = np.concatenate(
            [
                np.ones(samples // 2),
                np.full(samples // 2, 2.0),
            ]
        )
        df = pl.DataFrame(
            {
                "laps": lap_ids,
                "laptime": np.linspace(0.0, 12.0, samples),
                "Vmin": np.linspace(3.82, 3.62, samples),
                "Current": np.linspace(-10.0, 80.0, samples),
            }
        )

        fig, kpis = weakest_cell_fig({"synthetic": df})

        self.assertEqual(kpis["warnings"], [])
        self.assertTrue(
            any(shape.y0 == 3.75 and shape.y1 == 3.75 for shape in fig.layout.shapes),
            "weakest_cell_fig must draw the discharge floor at 3.75 V",
        )
        self.assertIn(
            "discharge floor 3.75 V",
            [annotation.text for annotation in fig.layout.annotations],
        )


if __name__ == "__main__":
    unittest.main()
