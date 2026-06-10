import polars as pl
import dynamics as dyn


def main():
    single = {"Cerpa": pl.read_csv("data/Cerpa_FSG.csv", infer_schema_length=2000)}
    multi = dict(single)
    multi["Martinez"] = pl.read_csv("data/Martinez_FSG.csv", infer_schema_length=2000)
    for label, dfs in (("single", single), ("multi", multi)):
        fig, kpis = dyn.brake_load_transfer_fig(dfs)
        assert fig.data, f"{label}: no traces"
        assert not kpis["warnings"], f"{label}: warnings {kpis['warnings']}"
        for run, v in kpis["runs"].items():
            assert 0.45 < v["fz_front_static"] < 0.55, (run, v["fz_front_static"])
            assert v["fz_front_at_peak"] > v["fz_front_static"], run
            assert v["samples"] > 500
            print(label, run, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    print("brake_load_transfer OK")


if __name__ == "__main__":
    main()
