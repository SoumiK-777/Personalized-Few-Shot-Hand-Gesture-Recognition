import os
import random
import logging
import time
import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# ==========================================
# CONFIGURATION & LOGGING SETUP
# ==========================================
# Adjust this BASE_DIR if you are running on Kaggle vs your Local Server
BASE_DIR = "/kaggle/input/datasets/soumikkumar/gestures"
DATA_DIR = os.path.join(BASE_DIR, "cropped_images")
LOG_FILE = "/kaggle/working/feat_training.log"

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
# MEMORY CLEANUP HELPER
# ==========================================
def free_gpu_memory(model=None):
    """Explicitly delete model, clear cache, and run GC to free GPU memory."""
    if model is not None:
        if isinstance(model, nn.DataParallel):
            del model.module
        del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info(
        f"GPU memory freed. "
        f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | "
        f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
    )

# ==========================================
# DISTANCE METRIC (FIXED)
# ==========================================
def cosine_dist(x, y):
    """
    Computes Scaled Cosine Similarity between query and adapted prototypes.
    x: (Q, D) - Query features
    y: (N, D) - Prototype features
    Returns: (Q, N) similarity matrix (values between -1.0 and +1.0)
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return torch.mm(x_norm, y_norm.t())

# ==========================================
# FEAT NETWORK CORE LOGIC
# ==========================================
class FEATNet(nn.Module):
    def __init__(self, backbone_name):
        super(FEATNet, self).__init__()
        logger.info(f"Initializing FEAT Backbone: {backbone_name}")
        
        # 1. Initialize base model (pooling features)
        base_backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'   
        )

        # 2. Extract dimension safely on CPU BEFORE wrapping in DataParallel
        dummy_x = torch.randn(2, 3, 224, 224) 
        base_backbone.eval()                    
        with torch.no_grad():
            self.feat_dim = base_backbone(dummy_x).shape[1]
        base_backbone.train()                   
        logger.info(f"Feature dimension: {self.feat_dim}")

        # 3. Multi-GPU Wrapper (Wrap ONLY the backbone to split heavy image processing)
        if torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(base_backbone)
        else:
            self.backbone = base_backbone

        # Calculate Transformer heads dynamically to ensure it divides the feat_dim
        heads = 1
        for h in range(8, 0, -1):
            if self.feat_dim % h == 0:
                heads = h
                break
        logger.info(f"Transformer heads: {heads}")

        # 4. Contextual Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim,
            nhead=heads,
            dim_feedforward=self.feat_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 5. Learnable Temperature Scalar
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def adapt_prototypes(self, support_x, support_y, n_way):
        support_features = self.backbone(support_x)
        initial_prototypes = []
        for i in range(n_way):
            mask = support_y == i
            initial_prototypes.append(support_features[mask].mean(0))
        
        initial_prototypes = torch.stack(initial_prototypes)  # (N, D)
        transformer_input = initial_prototypes.unsqueeze(0)   # (1, N, D)
        
        # Adapt prototypes by letting them attend to each other
        adapted_prototypes = self.transformer(transformer_input).squeeze(0)  # (N, D)
        return adapted_prototypes

    def forward_query(self, query_x):
        return self.backbone(query_x)

# ==========================================
# EPISODIC DATA SAMPLER
# ==========================================
def sample_episode(base_dir, split, n_way, k_shot, q_queries=10, transform=None):
    races = ['Black', 'Indian', 'White']
    race = random.choice(races)
    split_dir = os.path.join(base_dir, race, split)

    if not os.path.exists(split_dir):
        return None, None, None, None, race, []

    all_classes = [
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and d in ALLOWED_GESTURES
    ]

    if len(all_classes) < n_way:
        return None, None, None, None, race, []

    sampled_classes = random.sample(all_classes, n_way)

    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for label, cls in enumerate(sampled_classes):
        cls_dir = os.path.join(split_dir, cls)
        images = [
            os.path.join(cls_dir, img)
            for img in os.listdir(cls_dir)
            if img.endswith(('.jpg', '.png', '.jpeg'))
        ]
        needed_imgs = k_shot + q_queries
        if len(images) < needed_imgs:
            sampled_imgs = random.choices(images, k=needed_imgs)
        else:
            sampled_imgs = random.sample(images, needed_imgs)

        for img_path in sampled_imgs[:k_shot]:
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            support_images.append(img)
            support_labels.append(label)

        for img_path in sampled_imgs[k_shot:]:
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            query_images.append(img)
            query_labels.append(label)

    return (
        torch.stack(support_images),
        torch.stack(query_images),
        torch.tensor(support_labels, dtype=torch.long),
        torch.tensor(query_labels, dtype=torch.long),
        race,
        sampled_classes
    )

# ==========================================
# TRAINING LOGIC (FIXED)
# ==========================================
def train_episode(model, optimizer, support_x, support_y, query_x, query_y, n_way):
    model.train()
    optimizer.zero_grad()

    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    # 1. Adapt Support Features
    adapted_prototypes = model.adapt_prototypes(support_x, support_y, n_way)
    temperature = model.temperature

    # 2. Extract Query Features & Measure Inference
    start_inf = time.time()
    query_features = model.forward_query(query_x) 
    
    # --- MATH FIX ---
    # Calculate Cosine Similarity instead of Euclidean Distance
    similarities = cosine_dist(query_features, adapted_prototypes)
    
    # Scale similarities by temperature. NO negative sign because 
    # higher similarity = closer = higher probability
    logits = temperature * similarities 
    # ----------------
    end_inf = time.time()

    # 3. Cross Entropy / Log-Softmax Loss
    log_p_y = F.log_softmax(logits, dim=1)
    loss = -log_p_y.gather(1, query_y.unsqueeze(1)).squeeze().view(-1).mean()

    loss.backward()
    optimizer.step()

    # 4. Metrics
    _, predictions = log_p_y.max(1)
    correct = predictions.eq(query_y).sum().item()
    total_queries = query_y.size(0)
    accuracy = (correct / total_queries) * 100
    avg_inf_time_per_query = (end_inf - start_inf) / total_queries

    return loss.item(), accuracy, avg_inf_time_per_query

# ==========================================
# MAIN
# ==========================================
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

    model = None  

    for backbone_name in backbones:
        # Prevent Out Of Memory errors between architectures
        free_gpu_memory(model)
        model = None

        logger.info("=" * 90)
        logger.info(f"Starting Training Loop for FEAT | Backbone: {backbone_name}")
        logger.info("=" * 90)

        try:
            model = FEATNet(backbone_name).to(device)
            if torch.cuda.device_count() > 1:
                logger.info(f"Initialized across {torch.cuda.device_count()} GPUs!")
        except Exception as e:
            logger.error(f"Failed to load model {backbone_name}: {e}")
            free_gpu_memory(model)
            model = None
            continue

        # Differential Learning Rates: Protect pretrained backbone, train new Transformer faster
        backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
        head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': 1e-5}, 
            {'params': head_params, 'lr': 1e-3}       
        ])

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

                try:
                    loss, acc, inf_time_query = train_episode(
                        model, optimizer, s_x, s_y, q_x, q_y, n_way
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM at episode {episode+1}, skipping batch. {e}")
                        torch.cuda.empty_cache()
                        continue
                    raise

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
            logger.info(
                f"*** Completed {n_way}-way {k_shot}-shot. "
                f"Avg Loss: {avg_task_loss:.4f} | Avg Acc: {avg_task_acc:.2f}% ***\n"
            )

    free_gpu_memory(model)

if __name__ == "__main__":
    main()