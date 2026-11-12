"""Single source of truth: folder <-> Table-1 label <-> Round-2 paper values."""
import os
from pathlib import Path

BASE = Path(os.environ.get("CORAL_BASE", Path(__file__).resolve().parents[2]))
OUT = BASE / "_resub_v3"
R2_FEATURE_CACHE = BASE / "_uciqe_correction_audit" / "per_image_all_variants.csv"
R2_FROZEN_NPZ = BASE / "_corrected_qcs_full" / "corrected_frozen_cholesky.npz"

# label used in R2 cache  ->  (folder on disk, paper UIQM, UCIQE, AG, Ten, QCS)
# NOTE: R2 code scored the folder "Raw_Input"; the user designates "Input" as the raw folder.
#       A0/A1 checks confirm both hold identical images (feature match on a sample).
TABLE1 = {
    "Raw_Input":                 (("Input/Val", "Input/Test"), 28.93, 26.74, 15.08, 13947, 0.270),
    "Retinex_plain_w_phys=0":    ("Retinex_plain_w_phys=0",    30.16, 27.35, 18.50, 22278, 0.367),
    "Retinex_w_o_post":          ("Retinex_w_o_post",          31.71, 29.87, 22.38, 36737, 0.595),
    "Retinex_+_Default_MHF-UIE": ("Retinex_+_Default_MHF-UIE", 19.03, 23.11, 17.84, 14015, 0.149),
    "Retinex_+_Rule_based":      ("Retinex_+_Rule_based",      29.46, 30.42, 24.37, 48350, 0.725),
    "Retinex_+_LLaVA":           ("Retinex_+_LLaVA",           31.15, 30.15, 28.93, 63004, 0.815),
    "RAUNE_plain_w_phys=0":      ("RAUNE_plain_w_phys=0",      31.06, 29.48, 17.76, 19747, 0.386),
    "RAUNE_w_o_post":            ("RAUNE_w_o_post",            19.18, 21.34,  9.76,  7293, 0.096),
    "RAUNE_+_Default_MHF-UIE":   ("RAUNE_+_Default_MHF-UIE",   23.13, 25.19, 19.69, 22013, 0.290),
    "RAUNE_+_Rule_based":        ("RAUNE_+_Rule_based",        20.32, 23.16, 10.14, 10824, 0.188),
    "RAUNE_+_LLaVA":             ("RAUNE_+_LLaVA",             21.38, 23.11, 10.73, 11700, 0.194),
    "RAUNE_unsupervised":        ("RAUNE_unsupervised",        15.93, 26.21, 12.41, 13926, 0.298),
    "CCL_NET":                   ("CCL_NET",                   29.24, 30.00, 24.39, 29059, 0.457),
    "NUCE":                      ("NUCE",                      27.69, 27.49, 17.95, 17565, 0.298),
}
CONFIGS = list(TABLE1)
N_TEST = 4815
