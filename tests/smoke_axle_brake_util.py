import polars as pl
import dynamics as dyn


def main():
    single = {"Cerpa": pl.read_csv("data/Cerpa_FSG.csv", infer_schema_length=2000)}
    multi = dict(single)
    multi["Martinez"] = pl.read_csv("data/Martinez_FSG.csv", infer_schema_length=2000)
    for label, dfs in (("single", single), ("multi", multi)):
        fig, kpis = dyn.axle_brake_utilisation_fig(dfs)
        assert fig.data, f"{label}: no traces"
        assert not kpis["warnings"], f"{label}: warnings {kpis['warnings']}"
        for run, v in kpis["runs"].items():
            assert v["limiting_axle"] in ("front", "rear"), run
            assert 0.0 < v["util_front_median"] < 2.0, (run, v["util_front_median"])
            assert v["samples"] > 500
            print(label, run, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    print("axle_brake_util OK")


if __name__ == "__main__":
    main()
