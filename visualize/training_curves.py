import glob,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd,matplotlib.pyplot as plt
from common.config import RESULTS_DIR,FIGURES_DIR
def main():
 files=glob.glob(str(RESULTS_DIR/"episode_logs/*.csv"));fig,ax=plt.subplots(1,2,figsize=(12,4))
 for f in files:
  d=pd.read_csv(f);label=f.rsplit("/",1)[-1].replace(".csv","");ax[0].plot(d.episode,d.loss,label=label);ax[1].plot(d.episode,d.acc,label=label)
 ax[0].set_title("Loss");ax[1].set_title("Accuracy");ax[1].legend(fontsize=5);plt.savefig(FIGURES_DIR/"training_curves.png",dpi=300,bbox_inches="tight")
if __name__=="__main__":main()
