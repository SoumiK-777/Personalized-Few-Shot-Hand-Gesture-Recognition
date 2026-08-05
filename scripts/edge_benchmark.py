import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# ==========================================
# 1. HARDWARE SANDBOX CONFIGURATION
# ==========================================
# Force PyTorch to respect simulated edge CPU limits (e.g., 4 cores for Jetson Nano)
torch.set_num_threads(4) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing Edge Benchmark on: {device}")
if torch.cuda.is_available():
    # Example: If your desktop GPU has 12GB of VRAM, and you only want to 
    # give PyTorch 2GB to simulate a Jetson Nano's tight constraints:
    # 2GB / 12GB = ~0.16
    torch.cuda.set_per_process_memory_fraction(0.125, 0)

BACKBONE_NAME = 'tf_efficientnetv2_s.in21k'
GESTURE_DIR = './cropped_images/Asian/train'
K_SHOT = 10
N_WAY = 7

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class ProtoNetDeployment(nn.Module):
    def __init__(self, backbone_name):
        super(ProtoNetDeployment, self).__init__()
        # Feature extractor using standard pretrained weights
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
    def forward(self, x):
        return self.backbone(x)

def euclidean_dist(x, y):
    """
    Computes Euclidean distance between query embeddings (x) and prototypes (y).
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ==========================================
# 3. EPISODIC DATA LOADER
# ==========================================
def load_support_and_query_sets(preprocess):
    """Splits local folders into K_SHOT support images and remaining query images."""
    if not os.path.exists(GESTURE_DIR):
        print(f"[!] Error: {GESTURE_DIR} not found.")
        sys.exit(1)
        
    classes = [d for d in os.listdir(GESTURE_DIR) if os.path.isdir(os.path.join(GESTURE_DIR, d))]
    
    if len(classes) < N_WAY:
        print(f"[!] Error: Need at least {N_WAY} classes in {GESTURE_DIR}. Found {len(classes)}.")
        sys.exit(1)
        
    target_classes = classes[:N_WAY]
    
    support_data = []
    query_data = []
    
    for label_idx, cls in enumerate(target_classes):
        cls_dir = os.path.join(GESTURE_DIR, cls)
        images = [img for img in os.listdir(cls_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        if len(images) <= K_SHOT:
            print(f"[!] Error: Class '{cls}' has {len(images)} images. Need >{K_SHOT} to have queries left over.")
            sys.exit(1)
            
        # First K_SHOT for support
        for img_name in images[:K_SHOT]:
            img_path = os.path.join(cls_dir, img_name)
            pil_img = Image.open(img_path).convert('RGB')
            support_data.append((preprocess(pil_img), label_idx))
            
        # Remaining for query
        for img_name in images[K_SHOT:]:
            img_path = os.path.join(cls_dir, img_name)
            pil_img = Image.open(img_path).convert('RGB')
            query_data.append((preprocess(pil_img), label_idx))
            
    return support_data, query_data, target_classes

# ==========================================
# 4. BENCHMARK EXECUTION
# ==========================================
def run_benchmark():
    preprocess = get_transforms()
    model = ProtoNetDeployment(BACKBONE_NAME).to(device)
    model.eval()
    
    print("\n" + "="*50)
    print(f"LOADING DATA ({N_WAY}-WAY, {K_SHOT}-SHOT)")
    print("="*50)
    
    support_data, query_data, class_names = load_support_and_query_sets(preprocess)
    
    # ------------------------------------------
    # Phase A: Build Prototypes
    # ------------------------------------------
    print("\n[*] Phase A: Generating Prototypes...")
    prototypes = []
    
    with torch.no_grad():
        for i in range(N_WAY):
            # Extract support tensors for the current class
            cls_tensors = torch.stack([item[0] for item in support_data if item[1] == i]).to(device)
            features = model(cls_tensors)
            prototypes.append(features.mean(0))
            
        prototype_tensor = torch.stack(prototypes) # Shape: (N_WAY, D)
    print(f"[+] Successfully cached {N_WAY} prototypes.")
    
    # ------------------------------------------
    # Phase B: Sequential Inference (Latency Simulation)
    # ------------------------------------------
    print("\n[*] Phase B: Running Sequential Query Evaluation...")
    total_time = 0.0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for q_tensor, q_label in query_data:
            q_tensor = q_tensor.unsqueeze(0).to(device) # Batch size 1 to simulate live stream
            
            # Start Latency Timer
            start_time = time.time()
            
            # Extract feature and compute negative distance (logits)
            query_feature = model(q_tensor)
            dists = euclidean_dist(query_feature, prototype_tensor)
            logits = -dists # Prototypical networks use negative distance as logits
            
            # End Latency Timer
            end_time = time.time()
            total_time += (end_time - start_time)
            
            all_logits.append(logits.squeeze(0))
            all_labels.append(q_label)
            
    # ------------------------------------------
    # Phase C: Metrics Calculation
    # ------------------------------------------
    # Stack results
    stacked_logits = torch.stack(all_logits)
    stacked_labels = torch.tensor(all_labels, dtype=torch.long).to(device)
    
    # Calculate Loss (CrossEntropyLoss automatically applies log_softmax to the negative distances)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(stacked_logits, stacked_labels)
    
    # Calculate Accuracy
    _, predictions = torch.max(stacked_logits, dim=1)
    correct = (predictions == stacked_labels).sum().item()
    total_queries = len(query_data)
    accuracy = (correct / total_queries) * 100.0
    
    # Calculate Average Latency
    avg_latency_ms = (total_time / total_queries) * 1000.0
    
    # ------------------------------------------
    # Print Final Results
    # ------------------------------------------
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Queries Evaluated : {total_queries}")
    print(f"Accuracy                : {accuracy:.2f}%")
    print(f"Task Loss               : {loss.item():.4f}")
    print(f"Avg Inference Latency   : {avg_latency_ms:.2f} ms / query")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmark()