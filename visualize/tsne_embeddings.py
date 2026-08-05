import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np,matplotlib.pyplot as plt,torch
from sklearn.manifold import TSNE
from common.config import DATA_DIR,CKPT_DIR,FIGURES_DIR
from common.data import sample_episode
from proto_net import ProtoNet,TRANSFORM,device
from feat import FEATNet,load_feat_state_dict
def plot(method,model):
 xs=[];classes=[];races=[];model.eval()
 with torch.no_grad():
  for _ in range(20):
   sx,qx,sy,qy,race,names=sample_episode(DATA_DIR,"test",5,5,10,TRANSFORM)
   if sx is None:continue
   feature=model(qx.to(device)) if method=="protonet" else model.backbone(qx.to(device));xs.append(feature.cpu());classes += [names[i] for i in qy.tolist()];races += [race]*len(qy)
 z=TSNE(n_components=2,random_state=42).fit_transform(torch.cat(xs).numpy())
 for label,vals,file in [("class",classes,f"tsne_{method}_gesture.png"),("race",races,f"tsne_{method}_race.png")]:
  plt.figure();[plt.scatter(z[np.array(vals)==v,0],z[np.array(vals)==v,1],s=8,label=v) for v in sorted(set(vals))];plt.legend(fontsize=6);plt.title(f"{method} colored by {label}");plt.savefig(FIGURES_DIR/file,dpi=300,bbox_inches="tight");plt.close()
def main():
 p=ProtoNet("tf_efficientnetv2_s.in21k").to(device);p.load_state_dict(torch.load(CKPT_DIR/"protonet_tf_efficientnetv2_s.in21k.pt",map_location=device,weights_only=False));plot("protonet",p)
 f=FEATNet("tf_efficientnetv2_s.in21k").to(device);s=torch.load(CKPT_DIR/"feat_tf_efficientnetv2_s.in21k.pt",map_location=device,weights_only=False);load_feat_state_dict(f,s);plot("feat",f)
if __name__=="__main__":main()
