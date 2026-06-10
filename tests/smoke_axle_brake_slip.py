import polars as pl
import dynamics as dyn


def main():
    single = {"Cerpa": pl.read_csv("data/Cerpa_FSG.csv", infer_schema_length=2000)}
    multi = dict(single)
    multi["Martinez"] = pl.read_csv("data/Martinez_FSG.csv", infer_schema_length=2000)
    for label, dfs in (("single", single), ("multi", multi)):
        fig, kpis = dyn.axle_brake_slip_fig(dfs)
        assert fig.data, f"{label}: no traces"
        assert not kpis["warnings"], f"{label}: warnings {kpis['warnings']}"
        for run, v in kpis["runs"].items():
            assert v["axle_nearer_lockup"] in ("front", "rear"), run
            assert v["sr_front_p5"] <= 0.05, (run, v["sr_front_p5"])
            assert v["samples"] > 500
            print(label, run, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
    print("axle_brake_slip OK")


if __name__ == "__main__":
    main()
