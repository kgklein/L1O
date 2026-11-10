"""Offline event workflows. Scientific calculations live in independent modules."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .arrivals import build_arrival_table, accepted_arrivals
from .data import load_event_data
from .event import load_event, spacecraft_key
from .geometry import positions_at_arrivals, gse_rotation_estimate
from .reporting import provenance, write_json, write_csv, fit_payload, residual_table
from .timing import fit_timing, leave_one_out
from .uncertainty import monte_carlo_timing
from .visualization import plot_inspection, plot_geometry, plot_timing_diagnostics, save_figure

log = logging.getLogger(__name__)

ASSUMPTIONS = [
    "Single planar front, constant positive speed and normal over all crossings",
    "Propagation normal n=q/|q| with t-t_ref=a+q dot (r-r_ref); no sign reversal",
    "Each position evaluated at its own physical crossing epoch; linear interpolation, no extrapolation",
    "GSE axes approximated as fixed across the crossing interval; annual-rotation scale is diagnostic only",
    "Timing sigmas are absolute 1-sigma errors; independent Gaussian errors unless full covariance is supplied",
    "No ephemeris error covariance, clock calibration, ramp-selection bias, or front evolution modeled",
    "Six spacecraft give only two residual dof; residuals cannot identify curvature or rippling uniquely",
    "Native averaged measurement epochs preserved; no resampled magnetic products used",
]


def run_event(event_path, *, operation="inspect", manifest=None, outdir=None, spacecraft=None,
              zoom_s=120, relative=True) -> dict:
    """Run inspection or a validated fit, writing event-specific diagnostics.

    No network requests and no event/input-file mutations. An unsuccessful fit
    writes a status=failed shock_timing.json, never an apparent solution.
    All event members are required for fitting unless a subset is explicit.
    """
    if operation not in ("inspect", "fit"):
        raise ValueError("operation must be inspect or fit")
    event = load_event(event_path)
    output = Path(outdir).resolve() if outdir is not None else event.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {"event_id": event.event_id, "status": "failed", "operation": operation}
    try:
        log.info("Stage 1A: load %s native inputs, %s to %s", event.event_id, event.start, event.end)
        data = load_event_data(event, manifest)
        report.update(provenance(event, data), assumptions=ASSUMPTIONS)
        log.info("Stage 1B: detect proposals and retain configured manual selections")
        arrivals, detections = build_arrival_table(event, data.magnetic)
        write_csv(output / "arrivals.csv", arrivals)
        report.update(arrivals=arrivals.to_dict("records"), detections=detections)
        if operation == "inspect":
            times, kinds, positions, geometry_info, errors = {}, {}, [], {}, {}
            for row in arrivals.to_dict("records"):
                name = row["spacecraft"]
                time = row["crossing_time"] or (event.start + (event.end - event.start) / 2)
                times[name] = time
                kinds[name] = "accepted crossing" if row["accepted"] else "proposed crossing" if row["crossing_time"] else "inspection midpoint (no crossing)"
                try:
                    r, meta = positions_at_arrivals(data.ephemerides, {name: time}, event.inputs.get("ephemeris_max_gap_s", 900))
                    positions.append(r[0])
                    geometry_info[name] = meta[name]
                except ValueError as exc:
                    errors[name] = str(exc)
                    log.warning("%s inspection position unavailable: %s", name, exc)
            if positions:
                names = list(geometry_info)
                save_figure(plot_geometry(names, positions,
                    title="Inspection GSE geometry: own proposed/accepted epochs; midpoint if unselected"), output / "shock_geometry.png")
            report.update(status="inspection_only", geometry={"interpolation": geometry_info, "epoch_kind": kinds, "errors": errors})
            _inspection_figures(event, data.magnetic, arrivals, detections, output, zoom_s, relative)
            write_json(output / "inspection.json", report)
            return report
        names = list(event.spacecraft) if spacecraft is None else [spacecraft_key(n) for n in spacecraft]
        if len(set(names)) != len(names) or not set(names).issubset(event.spacecraft):
            raise ValueError("Fit subset must contain unique event spacecraft")
        selected = accepted_arrivals(arrivals, names)
        names = selected.spacecraft.tolist()
        times = selected.crossing_time.tolist()
        # Partial uncertainty vectors are never silently promoted to unweighted fits.
        supplied = selected.uncertainty_s.notna()
        if supplied.any() and not supplied.all():
            raise ValueError("Supply uncertainties for all fit members or omit all for an unweighted fit")
        sigma = selected.uncertainty_s.to_numpy(float) if supplied.all() else None
        positions, interpolation = positions_at_arrivals(data.ephemerides, dict(zip(names, times)),
                                                       event.inputs.get("ephemeris_max_gap_s", 900))
        options = event.fit.copy()
        covariance = options.get("covariance_s2")
        if covariance is not None:
            covariance = np.asarray(covariance, float)
            if covariance.shape != (len(event.spacecraft), len(event.spacecraft)):
                raise ValueError("fit.covariance_s2 must follow the complete event spacecraft order")
            indices = [list(event.spacecraft).index(n) for n in names]
            options["covariance_s2"] = covariance[np.ix_(indices, indices)]
        log.info("Stage 1C: affine timing inversion of %d accepted arrivals", len(names))
        fit = fit_timing(positions, times, sigma, spacecraft=names, **options)
        log.info("Fit speed=%.6g km/s, normal GSE=%s, geometry condition=%.6g, RMS=%.6g s",
                 fit.speed_km_s, fit.normal_gse, fit.geometry_condition, fit.rms_residual_s)
        loo = leave_one_out(positions, times, sigma, spacecraft=names, full_fit=fit, **options) if len(names) >= 5 else []
        mc_options = event.monte_carlo.copy()
        mc = None
        if fit.covariance_s2 is not None:
            mc = monte_carlo_timing(positions, times, sigma, spacecraft=names,
                ephemerides=data.ephemerides, ephemeris_max_gap_s=event.inputs.get("ephemeris_max_gap_s", 900),
                **options, **mc_options)
        else:
            fit.flags.append("No timing uncertainties supplied: Monte Carlo and weighted goodness-of-fit unavailable")
        rotation = gse_rotation_estimate(positions, (max(fit.observed_times) - min(fit.observed_times)).total_seconds(), fit.speed_km_s)
        if fit.uncertainties_s is not None and np.min(fit.uncertainties_s) < 1:
            fit.flags.append(f"Sub-second timing precision claimed: fixed-GSE approximation timing scale {rotation['timing_scale_s']:.3g} s; verify epoch convention and rotating-frame effects")
        if any(r["status"] == "failed" for r in loo):
            fit.flags.append("Some leave-one-out subsets failed; examine leave_one_out.csv")
        report.update(status="computed", fit=fit_payload(fit), geometry={"interpolation": interpolation,
                      "positions_km": dict(zip(names, positions)), "gse_rotation_estimate": rotation},
                      leave_one_out=loo, monte_carlo=mc)
        if report["fit"]["status"] == "flagged" or (mc is not None and mc["status"] != "ok") or any(r["status"] == "failed" for r in loo):
            report["status"] = "flagged"
        write_csv(output / "timing_residuals.csv", residual_table(fit))
        write_csv(output / "leave_one_out.csv", pd.DataFrame(loo) if loo else pd.DataFrame(columns=["excluded_spacecraft", "status"]))
        if mc is not None:
            write_csv(output / "monte_carlo_draws.csv", mc["draws"])
            report["monte_carlo"] = {k: v for k, v in mc.items() if k != "draws"}
            report["monte_carlo"]["draws_file"] = "monte_carlo_draws.csv"
        _inspection_figures(event, data.magnetic, arrivals, detections, output, zoom_s, relative)
        save_figure(plot_geometry(names, positions, fit=fit), output / "shock_geometry.png")
        save_figure(plot_timing_diagnostics(fit, loo), output / "timing_diagnostics.png")
        write_json(output / "shock_timing.json", report)
        return report
    except (ValueError, TypeError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
        report.update(status="failed", error=str(exc))
        if operation == "fit":
            # Remove an earlier solution from the current report if later work failed.
            report.pop("fit", None)
            write_json(output / "shock_timing.json", report)
        raise


def _inspection_figures(event, magnetic, arrivals, detections, output, zoom_s, relative):
    candidates = {n: d["candidates"] for n, d in detections.items()}
    common = dict(arrivals=arrivals, candidates=candidates, max_gap_s=event.detection.get("max_gap_s"))
    for stem, options in (("shock_inspection", {"window": (event.start, event.end)}),
                          ("shock_components", {"window": (event.start, event.end), "components": True}),
                          ("shock_zoom", {"zoom_s": zoom_s, "components": True})):
        save_figure(plot_inspection(magnetic, title=f"{event.event_id} | {event.start.date()} | native data", **common, **options),
                    output / f"{stem}.png")
    if relative:
        save_figure(plot_inspection(magnetic, relative=True, zoom_s=zoom_s, **common,
                                   title=f"{event.event_id} | profiles relative to own proposed/accepted crossings"), output / "shock_relative.png")
