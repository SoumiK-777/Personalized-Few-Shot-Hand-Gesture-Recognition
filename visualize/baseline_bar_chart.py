import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd,seaborn as sns,matplotlib.pyplot as plt
from common.config import RESULTS_DIR,FIGURES_DIR
def main():
 d=pd.read_csv(RESULTS_DIR/"all_results.csv");d=d[d.split=="test"];sns.barplot(data=d,x="method",y="acc_mean");plt.xticks(rotation=45,ha="right");plt.savefig(FIGURES_DIR/"baseline_bar_chart.png",dpi=300,bbox_inches="tight")
if __name__=="__main__":main()
