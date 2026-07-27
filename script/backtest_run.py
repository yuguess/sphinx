import argparse
import sys
from pathlib import Path

from sphinx.backtest import config
from sphinx.backtest import io
from sphinx.backtest import engine
from sphinx.backtest import holdings
from sphinx.backtest import results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清晰版原生数量回测。代码会显式检查时间对齐、容量约束和成本口径。")
    parser.add_argument("config", type=Path)
    parser.add_argument("-j", "--jobs", type=int, default=None, help="parallel worker count for per-day backtest")
    return parser.parse_args()


def default_jobs_for_mode(ep_holding_mode: str) -> int:
    if ep_holding_mode == "qty":
        return -1
    elif ep_holding_mode == "pct":
        return 1
    else:
        raise SystemExit(f"unsupported ep_holding_mode: {ep_holding_mode!r}")


def main():
    args = parse_args()
    conf = config.load_config(args.config)
    config.configure_environment(conf)
    settings = config.backtest_settings(conf)

    dates = io.selected_dates(conf)
    if not dates:
        raise SystemExit("no selected dates")

    jobs = args.jobs if getattr(args, "jobs", None) is not None else default_jobs_for_mode(settings.ep_holding_mode)

    if settings.ep_holding_mode == "qty":
        result = engine.run_backtest_clear(
            config=conf, dates=dates, ep_holding_univ_name=settings.ep_holding_univ_name,
            ep_holding_key=settings.ep_holding_key, diagnostics_dir=settings.diagnostics_dir, jobs=jobs,
        )
    elif settings.ep_holding_mode == "pct":
        ep_holding_by_date = holdings.read_quantity_frames_from_pct(
            dates=dates, ep_holding_univ_name=settings.ep_holding_univ_name, ep_holding_key=settings.ep_holding_key, nav=float(conf["nav"]))
        result = engine.run_backtest_clear_from_frames(
            config=conf, dates=dates, ep_holding_by_date=ep_holding_by_date, diagnostics_dir=settings.diagnostics_dir, jobs=jobs
        )
    else:
        raise SystemExit(f"unsupported ep_holding_mode: {settings.ep_holding_mode!r}")

    results.write_outputs(result, settings.output_dir)
    plot_path = None
    if settings.plot:
        plot_path = results.write_plot(result, settings.output_dir, settings.plot_path)

    print(f"ep_holding_univ_name {settings.ep_holding_univ_name}")
    print(f"ep_holding_key {settings.ep_holding_key}")
    print(f"output_dir {settings.output_dir}")
    if plot_path is not None:
        print(f"plot_path {plot_path}")
    print(result.metrics.to_string())
    # if settings.compare_dir:
    #     _results.compare(result, settings.compare_dir)

if __name__ == "__main__":
    main()
