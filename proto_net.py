"""Prototypical Network training/evaluation for race-stratified hand gesture episodes."""
import os, time
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from common.config import DATA_DIR, BACKBONES, TASKS, EPISODES_PER_TASK, EVAL_EPISODES, AUTOSAVE_EVERY, QUERY_PER_CLASS, CKPT_DIR
from common.data import sample_episode
from common.utils import set_seed, result_exists, append_result, append_episode, latest_path, save_state, checkpoint_test_acc, maybe_data_parallel, portable_state_dict, load_portable_state_dict

METHOD = "protonet"; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

class ProtoNet(nn.Module):
    def __init__(self, backbone_name):
        super().__init__(); self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")
    def forward(self, x): return self.backbone(x)

def logits_for_episode(model, sx, sy, qx, n_way):
    support = model(sx); prototypes = torch.stack([support[sy == i].mean(0) for i in range(n_way)])
    start = time.perf_counter(); query = model(qx); logits = -torch.cdist(query, prototypes).pow(2); elapsed = time.perf_counter() - start
    return logits, elapsed / len(qx)

def train_episode(model, optimizer, sx, sy, qx, qy, n_way):
    model.train(); optimizer.zero_grad(); sx,sy,qx,qy = (v.to(device) for v in (sx,sy,qx,qy))
    logits, per_query = logits_for_episode(model,sx,sy,qx,n_way); loss=F.cross_entropy(logits,qy); loss.backward(); optimizer.step()
    return loss.item(), (logits.argmax(1)==qy).float().mean().item()*100, per_query

@torch.no_grad()
def evaluate(model, split="test", n_episodes=200, n_way=5, k_shot=1):
    model.eval(); losses=[]; accs=[]; times=[]
    for _ in range(n_episodes):
        sx,qx,sy,qy,*_ = sample_episode(DATA_DIR,split,n_way,k_shot,QUERY_PER_CLASS,TRANSFORM)
        if sx is None: continue
        sx,sy,qx,qy=(v.to(device) for v in (sx,sy,qx,qy)); logits,latency=logits_for_episode(model,sx,sy,qx,n_way)
        losses.append(F.cross_entropy(logits,qy).item()); accs.append((logits.argmax(1)==qy).float().mean().item()*100); times.append(latency*1000)
    return dict(loss_mean=float(np.mean(losses)) if losses else 0., acc_mean=float(np.mean(accs)) if accs else 0., acc_std=float(np.std(accs)) if accs else 0., inf_ms_mean=float(np.mean(times)) if times else 0., n_episodes=len(accs))

def _state(model, optimizer, episode, backbone, n_way, k_shot, test_acc):
    return {"model_state_dict":portable_state_dict(model),"optimizer_state_dict":optimizer.state_dict(),"episode":episode,"backbone":backbone,"n_way":n_way,"k_shot":k_shot,"test_acc":test_acc}

def main():
    set_seed(); episodes=int(os.getenv("EPISODES_PER_TASK",EPISODES_PER_TASK))
    for backbone in BACKBONES:
      for n_way,k_shot in TASKS:
        if result_exists(METHOD,backbone,n_way,k_shot): print("completed; skipping",backbone,n_way,k_shot); continue
        model=maybe_data_parallel(ProtoNet(backbone).to(device)); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); start=0
        latest=latest_path(METHOD,backbone,n_way,k_shot)
        if latest.exists():
            state=torch.load(latest,map_location=device,weights_only=False); load_portable_state_dict(model,state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"]); start=state["episode"]+1; print("resuming",latest,"at",start)
        for episode in range(start,episodes):
            sx,qx,sy,qy,*_=sample_episode(DATA_DIR,"train",n_way,k_shot,QUERY_PER_CLASS,TRANSFORM)
            if sx is None: raise RuntimeError(f"No valid {n_way}-way train episode found under {DATA_DIR}")
            loss,acc,latency=train_episode(model,optimizer,sx,sy,qx,qy,n_way); append_episode(METHOD,backbone,n_way,k_shot,episode,loss,acc,latency*1000)
            if (episode+1)%AUTOSAVE_EVERY==0: save_state(_state(model,optimizer,episode,backbone,n_way,k_shot,checkpoint_test_acc(latest)),METHOD,backbone,n_way,k_shot,episode,checkpoint_test_acc(latest))
        metrics=evaluate(model,"test",EVAL_EPISODES,n_way,k_shot); state=_state(model,optimizer,episodes-1,backbone,n_way,k_shot,metrics["acc_mean"])
        save_state(state,METHOD,backbone,n_way,k_shot,episodes-1,metrics["acc_mean"],best=True); save_state(state,METHOD,backbone,n_way,k_shot,episodes-1,metrics["acc_mean"])
        append_result(method=METHOD,backbone=backbone,n_way=n_way,k_shot=k_shot,split="test",**metrics); torch.save(portable_state_dict(model),CKPT_DIR/f"{METHOD}_{backbone}.pt")
        print(backbone,n_way,k_shot,metrics)
if __name__ == "__main__": main()
