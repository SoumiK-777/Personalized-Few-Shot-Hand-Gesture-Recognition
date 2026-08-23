"""Frozen EfficientNetV2-S nearest-centroid and support-set linear fine-tuning baselines."""
import time, itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, timm, torch
import torch.nn.functional as F
from torchvision import transforms
from common.config import DATA_DIR,TASKS,EVAL_EPISODES
from common.data import sample_episode
from common.utils import set_seed,append_result,result_exists
device=torch.device("cuda" if torch.cuda.is_available() else "cpu");BACKBONE="tf_efficientnetv2_s.in21k";T=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
@torch.no_grad()
def embed(model,x):return model(x.to(device))
def run(kind,n,k,steps=0,lr=0):
 method = kind+(f"_s{steps}_lr{lr}" if steps else "")
 if result_exists(method, BACKBONE, n, k):
  print(f"completed; skipping {method} | {n}-way {k}-shot")
  return
 print(f"running {method} | {n}-way {k}-shot")
 model=timm.create_model(BACKBONE,pretrained=True,num_classes=0,global_pool="avg").to(device).eval();rows=[]
 for _ in range(EVAL_EPISODES):
  sx,qx,sy,qy,*_=sample_episode(DATA_DIR,"test",n,k,10,T)
  if sx is None:continue
  sf,qf=embed(model,sx),embed(model,qx);sy,qy=sy.to(device),qy.to(device);start=time.perf_counter()
  if kind=="frozen_centroid": scores=-torch.cdist(qf,torch.stack([sf[sy==i].mean(0) for i in range(n)]))
  else:
   head=torch.nn.Linear(sf.shape[1],n).to(device);opt=torch.optim.Adam(head.parameters(),lr=lr)
   for _ in range(steps): opt.zero_grad();loss=F.cross_entropy(head(sf),sy);loss.backward();opt.step()
   scores=head(qf)
  latency=(time.perf_counter()-start)/len(qx)*1000;loss=F.cross_entropy(scores,qy).item();rows.append((loss,(scores.argmax(1)==qy).float().mean().item()*100,latency))
 if not rows: raise RuntimeError(f"No valid test episodes for {method} | {n}-way {k}-shot")
 a=np.asarray(rows);append_result(method=method,backbone=BACKBONE,n_way=n,k_shot=k,split="test",loss_mean=float(a[:,0].mean()),acc_mean=float(a[:,1].mean()),acc_std=float(a[:,1].std()),inf_ms_mean=float(a[:,2].mean()),n_episodes=len(rows))
 print(f"completed {method} | {n}-way {k}-shot")
def main():
 set_seed()
 for n,k in TASKS:
  run("frozen_centroid",n,k)
  for s,lr in itertools.product((20,50,100),(1e-3,1e-2)):run("linear_finetune",n,k,s,lr)
if __name__=="__main__":main()
