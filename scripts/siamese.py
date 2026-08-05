import os
import random
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

# ==========================================
# CONFIGURATION & LOGGING SETUP
# ==========================================
BASE_DIR = "/home/soumik/soumik/misc"
DATA_DIR = os.path.join(BASE_DIR, "cropped_images")
LOG_FILE = os.path.join(BASE_DIR, "siamese_training.log")

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

# The specific gestures to be used
ALLOWED_GESTURES = [
    "four", "like", "one", "peace_inverted", "stop", "three", "three_gun",
    "dislike", "grabbing", "little_finger", "palm", "stop_inverted", "three2", 
    "thumb_index", "two_up", "fist", "grip", "middle_finger", "ok", "peace", 
    "rock", "three3", "thumb_index2", "two_up_inverted"
]

# ==========================================
# SIAMESE NETWORK & LOSS
# ==========================================
class SiameseNetwork(nn.Module):
    def __init__(self, backbone_name):
        super(SiameseNetwork, self).__init__()
        logger.info(f"Initializing Backbone: {backbone_name}")
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
    def forward(self, x1, x2):
        feat1 = self.backbone(x1)
        feat2 = self.backbone(x2)
        return feat1, feat2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        euclidean_distance = F.pairwise_distance(feat1, feat2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# ==========================================
# EPISODIC DATA SAMPLER
# ==========================================
def sample_episode(base_dir, split, n_way, k_shot, q_queries=10):
    races = ['Black', 'Indian', 'White']
    race = random.choice(races)
    split_dir = os.path.join(base_dir, race, split)
    
    if not os.path.exists(split_dir):
        logger.warning(f"Path not found: {split_dir}")
        return [], [], race, []
        
    # Filter classes to ONLY include those in ALLOWED_GESTURES
    all_classes = [d for d in os.listdir(split_dir) 
                   if os.path.isdir(os.path.join(split_dir, d)) and d in ALLOWED_GESTURES]
    
    if len(all_classes) < n_way:
        logger.warning(f"Not enough valid classes in {split_dir} for {n_way}-way task. Skipping episode.")
        return [], [], race, []

    sampled_classes = random.sample(all_classes, n_way)
    
    support_set = []
    query_set = []

    for label, cls in enumerate(sampled_classes):
        cls_dir = os.path.join(split_dir, cls)
        images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        needed_imgs = k_shot + q_queries
        if len(images) < needed_imgs:
            sampled_imgs = random.choices(images, k=needed_imgs)
        else:
            sampled_imgs = random.sample(images, needed_imgs)
            
        support_imgs = sampled_imgs[:k_shot]
        query_imgs = sampled_imgs[k_shot:]
        
        for img in support_imgs: support_set.append((img, label))
        for img in query_imgs: query_set.append((img, label))

    pairs = []
    labels = []
    # Create query pairs
    for q_img, q_label in query_set:
        for s_img, s_label in support_set:
            pairs.append((q_img, s_img))
            labels.append(1.0 if q_label == s_label else 0.0)

    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)

    return pairs, labels, race, sampled_classes

class SiameseEpisodeDataset(Dataset):
    def __init__(self, pairs, labels, transform):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_episode(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct_pairs = 0
    total_pairs = 0
    
    total_inference_time = 0.0
    
    for img1, img2, label in dataloader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        
        # --- Measure Inference/Forward Pass Time ---
        start_inf = time.time()
        feat1, feat2 = model(img1, img2)
        end_inf = time.time()
        
        total_inference_time += (end_inf - start_inf)
        # -------------------------------------------

        loss = criterion(feat1, feat2, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            distances = F.pairwise_distance(feat1, feat2)
            predictions = (distances < 1.0).float()
            correct_pairs += (predictions == label).sum().item()
            total_pairs += label.size(0)
            
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = (correct_pairs / total_pairs) * 100 if total_pairs > 0 else 0
    
    # Calculate average inference time per individual image pair
    avg_inf_time_per_pair = total_inference_time / total_pairs if total_pairs > 0 else 0
    
    return avg_loss, accuracy, avg_inf_time_per_pair

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
            model = SiameseNetwork(backbone_name).to(device)
        except Exception as e:
            logger.error(f"Failed to load model {backbone_name}: {e}")
            continue
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = ContrastiveLoss(margin=2.0)
        
        for n_way, k_shot in tasks:
            logger.info(f"--- Task Configuration: {n_way}-way {k_shot}-shot ---")
            
            task_loss = 0.0
            task_acc = 0.0
            valid_episodes = 0
            
            for episode in range(episodes_per_task):
                # Using 10 queries per class as requested
                pairs, labels, race, sampled_classes = sample_episode(DATA_DIR, 'train', n_way, k_shot, q_queries=10)
                
                if not pairs:
                    continue

                dataset = SiameseEpisodeDataset(pairs, labels, transform)
                # Ensure batch size handles the increased pair load well
                dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
                
                loss, acc, inf_time = train_episode(model, dataloader, optimizer, criterion)
                
                task_loss += loss
                task_acc += acc
                valid_episodes += 1
                
                # Output logs every 10 episodes
                if (episode + 1) % 10 == 0:
                    logger.info(
                        f"EP [{episode+1:02d}/{episodes_per_task}] | "
                        f"Race: {race:<7} | "
                        f"Loss: {loss:.4f} | "
                        f"Acc: {acc:>5.2f}% | "
                        f"Inf Time/Pair: {inf_time*1000:>5.2f}ms | " # Outputting in milliseconds
                        f"Classes: {sampled_classes}"
                    )
            
            avg_task_loss = task_loss / valid_episodes if valid_episodes > 0 else 0
            avg_task_acc = task_acc / valid_episodes if valid_episodes > 0 else 0
            logger.info(f"*** Completed {n_way}-way {k_shot}-shot. Avg Loss: {avg_task_loss:.4f} | Avg Acc: {avg_task_acc:.2f}% ***\n")

if __name__ == "__main__":
    main()