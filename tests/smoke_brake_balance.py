import polars as pl
import dynamics as dyn


def main():
    single = {"Cerpa": pl.read_csv("data/Cerpa_FSG.csv", infer_schema_length=2000)}
    multi = dict(single)
    multi["Martinez"] = pl.read_csv("data/Martinez_FSG.csv", infer_schema_length=2000)
    for label, dfs in (("single", single), ("multi", multi)):
        fig, kpis = dyn.brake_balance_fig(dfs)
        assert fig.data, f"{label}: no traces"
        assert not kpis["warnings"], f"{label}: warnings {kpis['warnings']}"
        for run, v in kpis["runs"].items():
            assert 0.4 < v["front_bias_median"] < 0.95, (run, v["front_bias_median"])
            assert 0.0 <= v["pct_rear_overbraked"] <= 100.0
            assert v["samples"] > 500
            print(label, run, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    print("brake_balance OK")


if __name__ == "__main__":
    main()
