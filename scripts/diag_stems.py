"""Quick diagnostic: how do file names in each calibration folder relate to Input/Val and Input/Test stems?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcs_eval import configs as C
from qcs_eval.io import list_images, stem

GROUPS = ["Retinex_plain_w_phys=0", "Retinex_w_o_post", "Retinex_+_Default_MHF-UIE", "Retinex_+_Rule_based",
          "RAUNE_plain_w_phys=0", "RAUNE_w_o_post", "RAUNE_+_Default_MHF-UIE", "RAUNE_+_Rule_based"]
val = {stem(p) for p in list_images(C.BASE / "Input" / "Val")}
test = {stem(p) for p in list_images(C.BASE / "Input" / "Test")}
print(f"Input/Val {len(val)} | Input/Test {len(test)} | e.g. {sorted(val)[:2]}")
for g in GROUPS:
    d = C.BASE / g
    fs = list_images(d, on_duplicate="keep")
    st = {stem(p) for p in fs}
    print(f"\n{g}: {len(fs)} files | in Val {len(st & val)} | in Test {len(st & test)} | in neither {len(st - val - test)}")
    print("   first names:", [p.name for p in fs[:3]])
    odd = sorted(st - val - test)[:3]
    if odd: print("   unmatched e.g.:", odd)
    subs = [x.name for x in d.iterdir() if x.is_dir()]
    if subs: print("   subfolders:", subs[:10])
