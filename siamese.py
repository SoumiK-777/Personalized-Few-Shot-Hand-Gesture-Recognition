"""Siamese episodic metric learning; balanced negative-pair subsampling prevents pair explosion."""
import os,time,random
import numpy as np,timm,torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from common.config import DATA_DIR,BACKBONES,TASKS,EPISODES_PER_TASK,EVAL_EPISODES,AUTOSAVE_EVERY,QUERY_PER_CLASS,CKPT_DIR
from common.data import sample_episode
from common.utils import set_seed,result_exists,append_result,append_episode,latest_path,save_state,checkpoint_test_acc,maybe_data_parallel,portable_state_dict,load_portable_state_dict
METHOD="siamese";device=torch.device("cuda" if torch.cuda.is_available() else "cpu");TRANSFORM=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class SiameseNetwork(nn.Module):
 def __init__(self,b):super().__init__();self.backbone=timm.create_model(b,pretrained=True,num_classes=0,global_pool="avg")
 def forward(self,x):return self.backbone(x)
def pair_loss_and_scores(model,sx,sy,qx,qy,n,training=False):
 sf=model(sx);qf=model(qx); positives=[];negatives=[]
 for qi,label in enumerate(qy.tolist()):
  for si,sl in enumerate(sy.tolist()): (positives if label==sl else negatives).append((qi,si,label==sl))
 # cap negatives to positives; this preserves a useful balanced signal instead of N-way cross-product growth
 if training: negatives=random.sample(negatives,min(len(negatives),len(positives)))
 pairs=positives+negatives; qi=torch.tensor([p[0] for p in pairs],device=device);si=torch.tensor([p[1] for p in pairs],device=device);target=torch.tensor([p[2] for p in pairs],device=device,dtype=torch.float)
 dist=F.pairwise_distance(qf[qi],sf[si]);loss=(target*dist.pow(2)+(1-target)*F.relu(2.-dist).pow(2)).mean()
 start=time.perf_counter();distances=torch.cdist(qf,sf);class_dist=torch.stack([distances[:,sy==i].mean(1) for i in range(n)],1);latency=(time.perf_counter()-start)/len(qx)
 return loss,-class_dist,latency
def train_episode(model,opt,sx,sy,qx,qy,n):
 model.train();opt.zero_grad();sx,sy,qx,qy=(x.to(device) for x in(sx,sy,qx,qy));loss,scores,t=pair_loss_and_scores(model,sx,sy,qx,qy,n,True);loss.backward();opt.step();return loss.item(),(scores.argmax(1)==qy).float().mean().item()*100,t
@torch.no_grad()
def evaluate(model,split="test",n_episodes=200,n_way=5,k_shot=1):
 model.eval();rows=[]
 for _ in range(n_episodes):
  sx,qx,sy,qy,*_=sample_episode(DATA_DIR,split,n_way,k_shot,QUERY_PER_CLASS,TRANSFORM)
  if sx is None:continue
  sx,sy,qx,qy=(x.to(device) for x in(sx,sy,qx,qy));loss,s,t=pair_loss_and_scores(model,sx,sy,qx,qy,n_way);rows.append((loss.item(),(s.argmax(1)==qy).float().mean().item()*100,t*1000))
 a=np.asarray(rows) if rows else np.zeros((1,3));return dict(loss_mean=float(a[:,0].mean()),acc_mean=float(a[:,1].mean()),acc_std=float(a[:,1].std()),inf_ms_mean=float(a[:,2].mean()),n_episodes=len(rows))
def main():
 set_seed();episodes=int(os.getenv("EPISODES_PER_TASK",EPISODES_PER_TASK))
 for b in BACKBONES:
  for n,k in TASKS:
   if result_exists(METHOD,b,n,k):print("completed; skipping",b,n,k);continue
   model=maybe_data_parallel(SiameseNetwork(b).to(device));opt=torch.optim.Adam(model.parameters(),lr=1e-3);latest=latest_path(METHOD,b,n,k);start=0
   if latest.exists():s=torch.load(latest,map_location=device,weights_only=False);load_portable_state_dict(model,s["model_state_dict"]);opt.load_state_dict(s["optimizer_state_dict"]);start=s["episode"]+1
   for ep in range(start,episodes):
    sx,qx,sy,qy,*_=sample_episode(DATA_DIR,"train",n,k,QUERY_PER_CLASS,TRANSFORM)
    if sx is None:raise RuntimeError("No valid training episode")
    loss,acc,t=train_episode(model,opt,sx,sy,qx,qy,n);append_episode(METHOD,b,n,k,ep,loss,acc,t*1000)
    if (ep+1)%AUTOSAVE_EVERY==0:save_state({"model_state_dict":portable_state_dict(model),"optimizer_state_dict":opt.state_dict(),"episode":ep,"backbone":b,"n_way":n,"k_shot":k,"test_acc":checkpoint_test_acc(latest)},METHOD,b,n,k,ep,checkpoint_test_acc(latest))
   m=evaluate(model,"test",EVAL_EPISODES,n,k);s={"model_state_dict":portable_state_dict(model),"optimizer_state_dict":opt.state_dict(),"episode":episodes-1,"backbone":b,"n_way":n,"k_shot":k,"test_acc":m["acc_mean"]};save_state(s,METHOD,b,n,k,episodes-1,m["acc_mean"],True);save_state(s,METHOD,b,n,k,episodes-1,m["acc_mean"]);append_result(method=METHOD,backbone=b,n_way=n,k_shot=k,split="test",**m);torch.save(portable_state_dict(model),CKPT_DIR/f"{METHOD}_{b}.pt")
if __name__=="__main__":main()
