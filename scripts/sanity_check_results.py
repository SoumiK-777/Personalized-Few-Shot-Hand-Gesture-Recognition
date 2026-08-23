import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import RESULTS_DIR
def main():
 output=RESULTS_DIR/"sanity_check_results.txt"
 if output.exists(): print(f"completed; skipping {output}"); return
 p=RESULTS_DIR/"all_results.csv"; rows=list(csv.DictReader(p.open())) if p.exists() else []; seen={}; collisions=[]
 for r in rows:
  key=(r["acc_mean"],r["acc_std"],r["inf_ms_mean"]); cfg=(r["method"],r["backbone"],r["n_way"],r["k_shot"])
  if key in seen and seen[key]!=cfg: collisions.append((seen[key],cfg,key))
  seen[key]=cfg
 lines=[f"SUSPICIOUS identical metrics: {c}" for c in collisions]
 lines.append(f"checked {len(rows)} rows; collisions={len(collisions)}")
 output.write_text("\n".join(lines)+"\n")
 print("\n".join(lines))
if __name__=="__main__":main()
