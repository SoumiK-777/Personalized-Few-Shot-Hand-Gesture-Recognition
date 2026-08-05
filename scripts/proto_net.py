import os
import random
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# ==========================================
# CONFIGURATION & LOGGING SETUP
# ==========================================
BASE_DIR = "/home/soumik/soumik/misc"
DATA_DIR = os.path.join(BASE_DIR, "cropped_images")
LOG_FILE = os.path.join(BASE_DIR, "protonet_training.log")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

ALLOWED_GESTURES = [
    "four", "like", "one", "peace_inverted", "stop", "three", "three_gun",
    "dislike", "grabbing", "little_finger", "palm", "stop_inverted", "three2", 
    "thumb_index", "two_up", "fist", "grip", "middle_finger", "ok", "peace", 
    "rock", "three3", "thumb_index2", "two_up_inverted"
]

# ==========================================
# PROTOTYPICAL NETWORK CORE LOGIC
# ==========================================
class ProtoNet(nn.Module):
    def __init__(self, backbone_name):
        super(ProtoNet, self).__init__()
        logger.info(f"Initializing Backbone: {backbone_name}")
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
    def forward(self, x):
        return self.backbone(x)

def euclidean_dist(x, y):
    """
    Computes Euclidean distance between two tensors.
    x: (N, D) - Query embeddings
    y: (M, D) - Prototype embeddings
    Returns: (N, M) distance matrix
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

# ==========================================
# EPISODIC DATA SAMPLER
# ==========================================
def sample_protonet_episode(base_dir, split, n_way, k_shot, q_queries=10, transform=None):
    """
    Returns pre-batched tensors for Support and Query sets for a single episode.
    """
    races = ['Black', 'Indian', 'White']
    race = random.choice(races)
    split_dir = os.path.join(base_dir, race, split)
    
    if not os.path.exists(split_dir):
        return None, None, None, None, race, []
        
    all_classes = [d for d in os.listdir(split_dir) 
                   if os.path.isdir(os.path.join(split_dir, d)) and d in ALLOWED_GESTURES]
    
    if len(all_classes) < n_way:
        return None, None, None, None, race, []

    sampled_classes = random.sample(all_classes, n_way)
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    for label, cls in enumerate(sampled_classes):
        cls_dir = os.path.join(split_dir, cls)
        images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        needed_imgs = k_shot + q_queries
        if len(images) < needed_imgs:
            sampled_imgs = random.choices(images, k=needed_imgs)
        else:
            sampled_imgs = random.sample(images, needed_imgs)
            
        # Process and transform images immediately
        for img_path in sampled_imgs[:k_shot]:
            img = Image.open(img_path).convert('RGB')
            if transform: img = transform(img)
            support_images.append(img)
            support_labels.append(label)
            
        for img_path in sampled_imgs[k_shot:]:
            img = Image.open(img_path).convert('RGB')
            if transform: img = transform(img)
            query_images.append(img)
            query_labels.append(label)

    # Stack into batched tensors
    support_tensors = torch.stack(support_images)
    query_tensors = torch.stack(query_images)
    support_labels = torch.tensor(support_labels, dtype=torch.long)
    query_labels = torch.tensor(query_labels, dtype=torch.long)

    return support_tensors, query_tensors, support_labels, query_labels, race, sampled_classes

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_episode(model, optimizer, support_x, support_y, query_x, query_y, n_way):
    model.train()
    optimizer.zero_grad()
    
    # Move to GPU
    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)
    
    # 1. Extract Support Features and calculate Prototypes
    support_features = model(support_x)
    
    prototypes = []
    for i in range(n_way):
        # Average the features of the K-shot support images for class i
        class_features = support_features[support_y == i]
        prototypes.append(class_features.mean(0))
    prototypes = torch.stack(prototypes)
    
    # 2. Extract Query Features & Measure Inference Time
    # (Measuring the time to process queries and compare them against prototypes)
    start_inf = time.time()
    
    query_features = model(query_x)
    dists = euclidean_dist(query_features, prototypes)
    
    end_inf = time.time()
    
    # 3. Calculate Loss and Accuracy (Based on Orobix official repo logic)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = -log_p_y.gather(1, query_y.unsqueeze(1)).squeeze().view(-1).mean()
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, y_hat = log_p_y.max(1)
    correct = y_hat.eq(query_y).sum().item()
    total_queries = query_y.size(0)
    accuracy = (correct / total_queries) * 100
    
    # Time metric: Total inference time divided by number of query images
    avg_inf_time_per_query = (end_inf - start_inf) / total_queries
    
    return loss.item(), accuracy, avg_inf_time_per_query

import gc  # Added for garbage collection

def main():
    backbones = [
        'mobilenetv4_conv_small.e2400_r224_in1k', 
        'tf_efficientnetv2_s.in21k',              
        'wide_resnet50_2'                         
    ]
    
    tasks = [
        (5, 5), (5, 10), (7, 5), (7, 10), (5, 1), (7, 1)
    ]
    
    episodes_per_task = 50 
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = None  # Initialize reference

    for backbone_name in backbones:
        # --- UNLOAD PREVIOUS MODEL ---
        if model is not None:
            logger.info(f"Unloading previous model to free VRAM...")
            del model
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2) # Brief pause to allow hardware to settle
        # -----------------------------

        logger.info("="*90)
        logger.info(f"Starting Training Loop for Backbone: {backbone_name}")
        logger.info("="*90)
        
        try:
            model = ProtoNet(backbone_name)
            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs!")
                model = nn.DataParallel(model)

            model.to(device)
        except Exception as e:
            logger.error(f"Failed to load model {backbone_name}: {e}")
            model = None # Ensure it stays None if it fails
            continue
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for n_way, k_shot in tasks:
            logger.info(f"--- Task Configuration: {n_way}-way {k_shot}-shot ---")
            
            task_loss = 0.0
            task_acc = 0.0
            valid_episodes = 0
            
            for episode in range(episodes_per_task):
                s_x, q_x, s_y, q_y, race, sampled_classes = sample_protonet_episode(
                    DATA_DIR, 'train', n_way, k_shot, q_queries=10, transform=transform
                )
                
                if s_x is None:
                    continue
                
                loss, acc, inf_time_query = train_episode(model, optimizer, s_x, s_y, q_x, q_y, n_way)
                
                task_loss += loss
                task_acc += acc
                valid_episodes += 1
                
                if (episode + 1) % 10 == 0:
                    logger.info(
                        f"EP [{episode+1:02d}/{episodes_per_task}] | "
                        f"Race: {race:<7} | "
                        f"Loss: {loss:.4f} | "
                        f"Acc: {acc:>5.2f}% | "
                        f"Inf Time/Query: {inf_time_query*1000:>5.2f}ms | " 
                        f"Classes: {sampled_classes}"
                    )
            
            avg_task_loss = task_loss / valid_episodes if valid_episodes > 0 else 0
            avg_task_acc = task_acc / valid_episodes if valid_episodes > 0 else 0
            logger.info(f"*** Completed {n_way}-way {k_shot}-shot. Avg Loss: {avg_task_loss:.4f} | Avg Acc: {avg_task_acc:.2f}% ***\n")
if __name__ == "__main__":
    main()