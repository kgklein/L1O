from __future__ import annotations

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
# We keep this minimal and flexible: prefer RTN vectors, else GSE vectors.
DATASET_LOGICAL_VARS = {
    # ACE
    "AC_H3_MFI": ["B_RTN", "B_GSE"],     # we can compute |B|
    "AC_H0_SWE": ["NP", "TP", "V_GSE"],  # often V is in GSE; we'll convert to pseudo-RTN later

    # Wind
    "WI_H0_MFI": ["B_GSE", "B_RTN"],
    "WI_PM_3DP": ["NP", "TP", "V_GSE"],

    # DSCOVR
    "DSCOVR_H0_MAG": ["B_GSE"],
    "DSCOVR_H1_FC":  ["NP", "TP", "V_GSE"],

    # SOHO
    "SOHO_CELIAS-PM_30S": ["NP", "TP", "V_GSE"],  # may not have all; we’ll handle gracefully
}
