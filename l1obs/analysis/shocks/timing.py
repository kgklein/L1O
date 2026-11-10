"""Affine, constant-speed planar timing inversion using scaled SVD least squares.

    t_i - t_ref = a + q · (r_i - r_ref) + epsilon_i

Positions are km, times seconds, q s/km. n=q/|q| points along propagation;
V=1/|q| is positive. There is no post-fit sign reversal.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from .event import positive, utc

log = logging.getLogger(__name__)


@dataclass
class TimingFit:
    spacecraft: tuple[str, ...]
    reference_time: pd.Timestamp
    reference_position_km: np.ndarray
    offset_s: float
    slowness_s_km: np.ndarray
    speed_km_s: float
    normal_gse: np.ndarray
    observed_times: pd.DatetimeIndex
    predicted_times: pd.DatetimeIndex
    observed_offsets_s: np.ndarray
    predicted_offsets_s: np.ndarray
    residuals_s: np.ndarray
    rms_residual_s: float
    max_abs_residual_s: float
    chi_square: float | None
    dof: int
    reduced_chi_square: float | None
    geometry_rank: int
    geometry_singular_values_km: np.ndarray
    geometry_condition: float
    design_condition: float
    method: str
    uncertainties_s: np.ndarray | None
    covariance_s2: np.ndarray | None
    parameter_covariance: np.ndarray | None
    flags: list[str]


def _noise_model(n, uncertainties_s=None, covariance_s2=None):
    sigma = None
    if uncertainties_s is not None:
        sigma = np.asarray(uncertainties_s, dtype=float)
        if sigma.shape != (n,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("All timing uncertainties must be finite, positive and have shape (n,)")
    covariance = None
    if covariance_s2 is not None:
        covariance = np.asarray(covariance_s2, dtype=float)
        if covariance.shape != (n, n) or not np.isfinite(covariance).all():
            raise ValueError("Timing covariance must be finite with shape (n, n)")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("Timing covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Timing covariance must be positive definite") from exc
        if sigma is not None and not np.allclose(sigma ** 2, np.diag(covariance), rtol=1e-8):
            raise ValueError("Uncertainties must match the covariance diagonal")
        sigma = np.sqrt(np.diag(covariance))
    elif sigma is not None:
        with np.errstate(over="ignore", under="ignore"):
            variances = sigma ** 2
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError("Timing variances are outside floating-point range")
        covariance = np.diag(variances)
    return sigma, covariance


def fit_timing(positions_km, times, uncertainties_s=None, *, spacecraft=None,
               method="auto", covariance_s2=None, condition_limit=1e6) -> TimingFit:
    """Fit every crossing with its own position and a freely fitted intercept.

    auto uses WLS/GLS when all uncertainties/covariance are supplied; otherwise
    OLS. Independent Gaussian 1-sigma errors are assumed unless covariance is
    supplied. Rank and conditioning use centered geometry and a dimensionless
    affine design (one common spatial scale, preserving geometric anisotropy).
    Rank-deficient or excessively ill-conditioned systems raise ValueError.
    """
    positions = np.asarray(positions_km, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 4 or not np.isfinite(positions).all():
        raise ValueError("At least four finite XYZ positions with shape (n, 3) are required")
    n = len(positions)
    observed = pd.DatetimeIndex([utc(t) for t in times]).as_unit("ns")
    if len(observed) != n:
        raise ValueError("Positions and crossing times must have equal length")
    names = tuple(str(i) for i in range(n)) if spacecraft is None else tuple(str(n) for n in spacecraft)
    if len(names) != n or len(set(names)) != n:
        raise ValueError("Spacecraft names must be unique with one per arrival")
    sigma, covariance = _noise_model(n, uncertainties_s, covariance_s2)
    if method not in ("auto", "weighted", "unweighted"):
        raise ValueError("method must be auto, weighted, or unweighted")
    weighted = covariance is not None if method == "auto" else method == "weighted"
    if weighted and covariance is None:
        raise ValueError("Weighted fit requires uncertainties or a timing covariance")
    condition_limit = positive(condition_limit, "condition_limit")
    reference_time = observed.min()
    offsets = (observed - reference_time).total_seconds().to_numpy()
    reference_position = positions.mean(axis=0)
    centered = positions - reference_position
    singular_geometry = np.linalg.svd(centered, compute_uv=False)
    tolerance = np.finfo(float).eps * max(centered.shape) * singular_geometry[0]
    geometry_rank = int(np.sum(singular_geometry > tolerance))
    if geometry_rank < 3:
        raise ValueError(f"Rank-deficient spacecraft geometry: rank {geometry_rank}, need 3")
    geometry_condition = float(singular_geometry[0] / singular_geometry[-1])
    if geometry_condition > condition_limit:
        raise ValueError(f"Ill-conditioned spacecraft geometry: {geometry_condition:g} exceeds {condition_limit:g}")
    length_scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    design = np.column_stack([np.ones(n), centered / length_scale])
    whiten = None if covariance is None else np.linalg.cholesky(covariance)
    A = np.linalg.solve(whiten, design) if weighted else design
    y = np.linalg.solve(whiten, offsets) if weighted else offsets
    U, singular, Vt = np.linalg.svd(A, full_matrices=False)
    rank = np.sum(singular > np.finfo(float).eps * max(A.shape) * singular[0])
    design_condition = float(singular[0] / singular[-1])
    if rank < 4 or design_condition > condition_limit:
        raise ValueError(f"Rank-deficient or ill-conditioned affine timing design: rank {rank}, condition {design_condition:g}")
    coefficients = Vt.T @ ((U.T @ y) / singular)
    q = coefficients[1:] / length_scale
    qnorm = float(np.linalg.norm(q))
    if not np.isfinite(qnorm):
        raise ValueError("Nonfinite slowness; no physical timing solution")
    if qnorm <= np.finfo(float).eps * max(1., np.linalg.norm(offsets)) / length_scale:
        raise ValueError("Zero or numerically unresolved slowness; finite speed cannot be inferred")
    speed = 1 / qnorm
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Shock speed is not finite and positive")
    normal = q / qnorm
    predicted_offsets = design @ coefficients
    residuals = offsets - predicted_offsets
    predicted = pd.DatetimeIndex(reference_time + pd.to_timedelta(predicted_offsets, unit="s"))
    dof = n - 4
    chi_square = None if whiten is None else float(np.sum(np.linalg.solve(whiten, residuals) ** 2))
    reduced = chi_square / dof if chi_square is not None and dof > 0 else None
    transform = np.diag([1., 1 / length_scale, 1 / length_scale, 1 / length_scale])
    # The sandwich formula also covers explicitly unweighted fits with known noise.
    estimator = (Vt.T / singular) @ U.T
    noise = np.eye(n) if weighted else covariance
    if noise is None and dof > 0:
        noise = np.eye(n) * np.sum(residuals ** 2) / dof
    parameter_covariance = None if noise is None else transform @ estimator @ noise @ estimator.T @ transform
    flags = []
    if dof <= 2:
        flags.append(f"Only {dof} residual degrees of freedom; limited model-consistency diagnostic")
    if geometry_condition > 1e3 or design_condition > 1e3:
        flags.append("Weak geometric/design conditioning (>1000)")
    if reduced is not None and reduced > 4:
        flags.append("Large residuals relative to stated timing errors (reduced chi-square >4); review planar model and arrivals")
    if not weighted and covariance is not None and not np.allclose(covariance, np.eye(n) * covariance[0, 0]):
        flags.append("OLS with heterogeneous/correlated errors: weighted residual statistic is not a calibrated chi-square test")
    if sigma is not None and np.max(np.abs(residuals) / sigma) > 3:
        flags.append("At least one timing residual exceeds 3 supplied standard deviations")
    log.debug("Timing fit %s: rank=%d, geometry condition=%.6g, design condition=%.6g, speed=%.6g km/s, n=%s",
              "weighted" if weighted else "unweighted", geometry_rank, geometry_condition, design_condition, speed, normal)
    return TimingFit(names, reference_time, reference_position, float(coefficients[0]), q, speed, normal,
                     observed, predicted, offsets, predicted_offsets, residuals,
                     float(np.sqrt(np.mean(residuals ** 2))), float(np.max(np.abs(residuals))),
                     chi_square, dof, reduced, geometry_rank, singular_geometry, geometry_condition,
                     design_condition, "weighted" if weighted else "unweighted", sigma, covariance,
                     parameter_covariance, flags)


def angular_separation_deg(normals, central_normal):
    normals = np.asarray(normals, dtype=float)
    central = np.asarray(central_normal, dtype=float)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    central = central / np.linalg.norm(central)
    return np.degrees(np.arccos(np.clip(normals @ central, -1, 1)))


def leave_one_out(positions_km, times, uncertainties_s=None, *, spacecraft=None, full_fit=None, **kwargs) -> list[dict]:
    """Return every subset fit, including failures instead of suppressing them."""
    positions = np.asarray(positions_km)
    times = list(times)
    full = full_fit or fit_timing(positions, times, uncertainties_s, spacecraft=spacecraft, **kwargs)
    if len(positions) < 5:
        raise ValueError("Leave-one-out requires at least five spacecraft")
    kwargs.setdefault("method", full.method)
    covariance = kwargs.pop("covariance_s2", full.covariance_s2)
    rows = []
    for excluded, name in enumerate(full.spacecraft):
        keep = np.arange(len(positions)) != excluded
        row = {"excluded_spacecraft": name, "status": "failed", "error": ""}
        try:
            subset_sigma = None if uncertainties_s is None else np.asarray(uncertainties_s)[keep]
            subset_covariance = None if covariance is None else np.asarray(covariance)[np.ix_(keep, keep)]
            fit = fit_timing(positions[keep], np.asarray(times)[keep], subset_sigma,
                             spacecraft=np.asarray(full.spacecraft)[keep], covariance_s2=subset_covariance, **kwargs)
            row.update(status="ok", speed_km_s=fit.speed_km_s, nx_gse=fit.normal_gse[0],
                       ny_gse=fit.normal_gse[1], nz_gse=fit.normal_gse[2],
                       angular_difference_deg=float(angular_separation_deg(fit.normal_gse, full.normal_gse)),
                       speed_change_km_s=fit.speed_km_s - full.speed_km_s,
                       geometry_condition=fit.geometry_condition, geometry_rank=fit.geometry_rank,
                       design_condition=fit.design_condition, rms_residual_s=fit.rms_residual_s,
                       residuals_s=dict(zip(fit.spacecraft, fit.residuals_s.tolist())),
                       dof=fit.dof, chi_square=fit.chi_square, reduced_chi_square=fit.reduced_chi_square,
                       flags=fit.flags)
        except (ValueError, np.linalg.LinAlgError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows
