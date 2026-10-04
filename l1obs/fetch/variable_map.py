from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MagneticProduct:
    time: str
    vector: str
    magnitude: str
    cadence_seconds: float
    quality: str | None = None

    def variables(self) -> dict[str, str]:
        variables = {"time": self.time, "B_GSE": self.vector, "B_MAG": self.magnitude}
        if self.quality is not None:
            variables["quality"] = self.quality
        return variables


MAGNETIC_PRODUCTS = {
    "WI_H0_MFI": MagneticProduct("Epoch3", "B3GSE", "B3F1", 3),
    "AC_H3_MFI": MagneticProduct("Epoch", "BGSEc", "Magnitude", 1),
    "DSCOVR_H0_MAG": MagneticProduct("Epoch1", "B1GSE", "B1F1", 1, "FLAG1"),
    "IMAP_MAG_L2_NORM-GSE": MagneticProduct("epoch", "b_gse", "magnitude", 0.5, "quality_flags"),
}

# Candidate names (ordered by preference). We will pick the first that exists.
# If none exist, we raise an error that prints available variables.

VAR_CANDIDATES = {
    # Magnetic field vector
    "B_RTN": [
        "BRTN", "B_RTN", "B_RTN_GSE", "B_RTN_GSM",  # some datasets
        "B_RTN_VEC", "B_RTN_VECTOR",
    ],
    "B_GSE": [
        "BGSE", "B_GSE", "B_GSE_VEC", "B_GSE_VECTOR",
        "B_x_GSE", "B_y_GSE", "B_z_GSE",
    ],
    # Velocity vector
    "V_GSE": [
        "VGSE", "V_GSE", "V_GSE_VEC", "V_GSE_VECTOR",
        "Vx_GSE", "Vy_GSE", "Vz_GSE",
    ],
    "V_RTN": [
        "VRTN", "V_RTN", "V_RTN_VEC", "V_RTN_VECTOR",
    ],
    # Scalars
    "NP": ["Np", "N_p", "DENSITY", "proton_density", "N_P"],
    "TP": ["Tp", "T_p", "TEMP", "proton_temperature", "T_P"],
}

# Per dataset, what "logical variables" we want.
# Magnetic datasets use the exact product specifications above; plasma
# datasets retain candidate-based resolution.
DATASET_LOGICAL_VARS = {
    # ACE
    "AC_H3_MFI": ["B_GSE", "B_MAG"],
    "AC_H0_SWE": ["NP", "TP", "V_GSE"],  # often V is in GSE; we'll convert to pseudo-RTN later

    # Wind
    "WI_H0_MFI": ["B_GSE", "B_MAG"],
    "WI_PM_3DP": ["NP", "TP", "V_GSE"],

    # DSCOVR
    "DSCOVR_H0_MAG": ["B_GSE", "B_MAG"],
    "DSCOVR_H1_FC":  ["NP", "TP", "V_GSE"],

    # IMAP
    "IMAP_MAG_L2_NORM-GSE": ["B_GSE", "B_MAG"],

    # SOHO
    "SOHO_CELIAS-PM_30S": ["NP", "TP", "V_GSE"],  # may not have all; we’ll handle gracefully
}
