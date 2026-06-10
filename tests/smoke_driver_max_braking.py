import polars as pl
import driver as drv
import dynamics as dyn


def main():
    single = {"Cerpa": pl.read_csv("data/Cerpa_FSG.csv", infer_schema_length=2000)}
    multi = dict(single)
    multi["Martinez"] = pl.read_csv("data/Martinez_FSG.csv", infer_schema_length=2000)
    assert not hasattr(dyn, "max_braking_g_per_lap_fig"), "still present in dynamics"
    for label, dfs in (("single", single), ("multi", multi)):
        fig, kpis = drv.max_braking_g_per_lap_fig(dfs)
        assert fig.data, f"{label}: no traces"
        for run, v in kpis["runs"].items():
            assert v["best_max_braking_g"] < 0.0, run
            assert v["valid_laps"] > 0, run
            print(label, run, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    print("driver_max_braking OK")


if __name__ == "__main__":
    main()
