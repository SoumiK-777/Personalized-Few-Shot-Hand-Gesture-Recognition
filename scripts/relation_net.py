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
LOG_FILE = os.path.join(BASE_DIR, "relationnet_training.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

ALLOWED_GESTURES = [
    "four", "like", "one", "peace_inverted", "stop", "three", "three_gun",
    "dislike", "grabbing", "little_finger", "palm", "stop_inverted", "three2", 
    "thumb_index", "two_up", "fist", "grip", "middle_finger", "ok", "peace", 
    "rock", "three3", "thumb_index2", "two_up_inverted"
]

# ==========================================
# RELATION NETWORK CORE LOGIC
# ==========================================
class RelationModule(nn.Module):
    """
    The neural network that learns to compare two concatenated feature vectors.
    Outputs a relation score between 0 (different) and 1 (same class).
    """
    def __init__(self, input_size, hidden_size=512):
        super(RelationModule, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = torch.sigmoid(self.fc_out(x))
        return out

# ==========================================
# EPISODIC DATA SAMPLER
# ==========================================
def sample_episode(base_dir, split, n_way, k_shot, q_queries=10, transform=None):
    """
    Returns pre-batched tensors for Support and Query sets.
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

    support_tensors = torch.stack(support_images)
    query_tensors = torch.stack(query_images)
    support_labels = torch.tensor(support_labels, dtype=torch.long)
    query_labels = torch.tensor(query_labels, dtype=torch.long)

    return support_tensors, query_tensors, support_labels, query_labels, race, sampled_classes

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_episode(embed_model, relation_model, optimizer, support_x, support_y, query_x, query_y, n_way):
    embed_model.train()
    relation_model.train()
    optimizer.zero_grad()
    
    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)
    
    # 1. Extract Support Features and form class prototypes (average of K shots)
    support_features = embed_model(support_x)
    
    class_features = []
    for i in range(n_way):
        class_feat = support_features[support_y == i].mean(0)
        class_features.append(class_feat)
    class_features = torch.stack(class_features) # Shape: (N, D)
    
    # ---------------------------------------------------
    # INFERENCE TIME MEASUREMENT START
    # (Extracting query features + Relation Module mapping)
    start_inf = time.time()
    
    # 2. Extract Query Features
    query_features = embed_model(query_x) # Shape: (Q, D)
    
    Q = query_features.size(0)
    N = class_features.size(0)
    D = class_features.size(1)
    
    # 3. Expand and Concatenate to form Query-Class pairs
    # q_ext shape: (Q, N, D) | c_ext shape: (Q, N, D)
    q_ext = query_features.unsqueeze(1).expand(Q, N, D)
    c_ext = class_features.unsqueeze(0).expand(Q, N, D)
    
    # Concatenate along the feature dimension -> Shape: (Q, N, 2D) -> Flatten for MLP: (Q*N, 2D)
    relation_pairs = torch.cat((c_ext, q_ext), dim=2).view(-1, D * 2)
    
    # 4. Pass through Relation Network to get scores
    relation_scores = relation_model(relation_pairs).view(Q, N)
    
    end_inf = time.time()
    # INFERENCE TIME MEASUREMENT END
    # ---------------------------------------------------

    # 5. MSE Loss Calculation
    # Ground truth: one-hot encoded matrix where 1 means same class, 0 means different
    targets = F.one_hot(query_y, num_classes=n_way).float().to(device)
    loss = F.mse_loss(relation_scores, targets)
    
    loss.backward()
    optimizer.step()
    
    # 6. Accuracy Calculation (Index with the highest relation score)
    _, predictions = relation_scores.max(1)
    correct = predictions.eq(query_y).sum().item()
    accuracy = (correct / Q) * 100
    
    # Time Metric
    avg_inf_time_per_query = (end_inf - start_inf) / Q
    
    return loss.item(), accuracy, avg_inf_time_per_query

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

    for backbone_name in backbones:
        logger.info("="*90)
        logger.info(f"Starting Training Loop for Backbone: {backbone_name}")
        logger.info("="*90)
        
        try:
            embed_model = timm.create_model(backbone_name, pretrained=True, num_classes=0).to(device)
            # Dynamically get the output dimension of the chosen backbone
            feature_dim = embed_model.num_features 
            
            # The Relation Module takes concatenated support and query features (hence feature_dim * 2)
            relation_model = RelationModule(input_size=feature_dim * 2).to(device)
            
        except Exception as e:
            logger.error(f"Failed to load model {backbone_name}: {e}")
            continue
            
        # We must optimize BOTH models together
        optimizer = torch.optim.Adam([
            {'params': embed_model.parameters()},
            {'params': relation_model.parameters()}
        ], lr=0.001)
        
        for n_way, k_shot in tasks:
            logger.info(f"--- Task Configuration: {n_way}-way {k_shot}-shot ---")
            
            task_loss = 0.0
            task_acc = 0.0
            valid_episodes = 0
            
            for episode in range(episodes_per_task):
                s_x, q_x, s_y, q_y, race, sampled_classes = sample_episode(
                    DATA_DIR, 'train', n_way, k_shot, q_queries=10, transform=transform
                )
                
                if s_x is None:
                    continue
                
                loss, acc, inf_time_query = train_episode(
                    embed_model, relation_model, optimizer, s_x, s_y, q_x, q_y, n_way
                )
                
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