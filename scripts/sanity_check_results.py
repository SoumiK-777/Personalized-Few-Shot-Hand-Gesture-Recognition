import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import RESULTS_DIR
def main():
 p=RESULTS_DIR/"all_results.csv"; rows=list(csv.DictReader(p.open())) if p.exists() else []; seen={}; collisions=[]
 for r in rows:
  key=(r["acc_mean"],r["acc_std"],r["inf_ms_mean"]); cfg=(r["method"],r["backbone"],r["n_way"],r["k_shot"])
  if key in seen and seen[key]!=cfg: collisions.append((seen[key],cfg,key))
  seen[key]=cfg
 for c in collisions: print("SUSPICIOUS identical metrics:",c)
 print(f"checked {len(rows)} rows; collisions={len(collisions)}")
if __name__=="__main__":main()
