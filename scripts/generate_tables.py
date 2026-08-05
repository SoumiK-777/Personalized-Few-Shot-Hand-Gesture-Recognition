import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from common.config import RESULTS_DIR
def main():
 p=RESULTS_DIR/"all_results.csv"; rows=list(csv.DictReader(p.open())) if p.exists() else []
 groups=defaultdict(list)
 for r in rows: groups[r["method"]].append(r)
 out=[]
 for method,rs in sorted(groups.items()):
  out += [f"% {method}","\\begin{tabular}{llrrrr}","Backbone & N & K & Loss & Accuracy & ms \\\\ \\hline"]
  out += [f"{r['backbone']} & {r['n_way']} & {r['k_shot']} & {float(r['loss_mean']):.4f} & {float(r['acc_mean']):.2f} $\\pm$ {float(r['acc_std']):.2f} & {float(r['inf_ms_mean']):.2f} \\\\" for r in rs]
  out += ["\\end{tabular}",""]
 (RESULTS_DIR/"tables_I_VII.tex").write_text("\n".join(out));print("wrote tables_I_VII.tex")
if __name__=="__main__":main()
