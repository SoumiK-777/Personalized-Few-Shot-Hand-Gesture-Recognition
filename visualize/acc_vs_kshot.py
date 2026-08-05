import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from common.config import RESULTS_DIR,FIGURES_DIR
def main():
 d=pd.read_csv(RESULTS_DIR/"all_results.csv");d=d[d.split=="test"];sns.lineplot(data=d,x="k_shot",y="acc_mean",hue="method",marker="o");plt.savefig(FIGURES_DIR/"acc_vs_kshot.png",dpi=300,bbox_inches="tight")
if __name__=="__main__":main()
