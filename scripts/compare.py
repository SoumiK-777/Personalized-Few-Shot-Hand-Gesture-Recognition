import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- CONFIGURATION ---
BASE_PATH = "cropped_images/Indian/train/"
CLASSES = ["four", "stop", "fist", "ok", "peace"]
COLORS = ['#2c3e50', '#8e44ad', '#2980b9', '#27ae60', '#d35400']
SHOTS = 4  # Number of images to display per cluster
ZOOM = 0.25 # Adjust based on your original image resolution (e.g. 0.15 to 0.3)

def get_random_images(class_name, count):
    """Fetches random image paths from a specific gesture folder."""
    folder_path = os.path.join(BASE_PATH, class_name)
    if not os.path.exists(folder_path):
        return []
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        return []
    sampled = random.choices(images, k=count) if len(images) < count else random.sample(images, count)
    return [os.path.join(folder_path, img) for img in sampled]

def add_image_marker(ax, img_path, xy, edge_color, zoom=0.2):
    """Helper to draw an image at a specific (x,y) coordinate."""
    try:
        img = plt.imread(img_path)
        imagebox = OffsetImage(img, zoom=zoom)
        # Put a colored box around the image corresponding to its class
        ab = AnnotationBbox(imagebox, xy, frameon=True, pad=0.1, 
                            bboxprops=dict(edgecolor=edge_color, lw=2, facecolor='none'))
        ax.add_artist(ab)
    except Exception as e:
        print(f"Could not load {img_path}: {e}")

def create_image_prototype_comparison():
    np.random.seed(42)

    # --- 1. DEFINE SYNTHETIC 2D LOCATIONS ---
    # We force the clusters to specific locations so the diagram looks clean
    base_centers = np.array([
        [-4, 4],   # Four (Top Left)
        [4, 4],    # Stop (Top Right)
        [4, -4],   # Fist (Bottom Right)
        [-4, -4],  # Ok (Bottom Left)
        [0, 0]     # Peace (Center)
    ])

    # Generate slightly scattered (x, y) coordinates for the actual images
    support_coords = []
    support_images = []
    
    for i, class_name in enumerate(CLASSES):
        imgs = get_random_images(class_name, SHOTS)
        support_images.append(imgs)
        # Add random scatter around the center for the images
        points = base_centers[i] + np.random.randn(SHOTS, 2) * 1.2
        support_coords.append(points)

    # --- 2. COMPUTE PROTOTYPES ---
    # ProtoNet: The mathematical mean of the scattered image coordinates
    proto_net_prototypes = np.array([pts.mean(axis=0) for pts in support_coords])

    # FEAT Simulation: Transformer pulls everything toward the global center of mass
    global_mean = proto_net_prototypes.mean(axis=0)
    collapse_factor = 0.7  # 70% pull towards the center
    
    feat_prototypes = []
    for p in proto_net_prototypes:
        adapted_p = p * (1 - collapse_factor) + global_mean * collapse_factor
        # Transformers also slightly scramble/shift features locally
        adapted_p += np.random.randn(2) * 0.5 
        feat_prototypes.append(adapted_p)
    feat_prototypes = np.array(feat_prototypes)


    # --- 3. VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Embedding Space: ProtoNet vs. FEAT Adaptation Failure", fontsize=20, fontweight='bold', y=0.98)

    def plot_support_images(ax):
        """Draws the actual dataset images at their assigned scattered coordinates."""
        for i, class_name in enumerate(CLASSES):
            coords = support_coords[i]
            imgs = support_images[i]
            for j in range(SHOTS):
                if j < len(imgs):
                    add_image_marker(ax, imgs[j], coords[j], COLORS[i], zoom=ZOOM)

    # --- SUBPLOT 1: PROTONET ---
    ax1.set_title("ProtoNet: Unadapted Prototypes\n(Accurate Local Means)", fontsize=16, pad=15)
    plot_support_images(ax1)
    
    for i, p in enumerate(proto_net_prototypes):
        # Draw the prototype center as a large, bright star
        ax1.scatter(p[0], p[1], color=COLORS[i], marker='*', s=800, edgecolor='black', linewidth=2, zorder=10)
        # Draw decision boundary radius
        circle = patches.Circle((p[0], p[1]), radius=2.2, color=COLORS[i], fill=False, linestyle='--', alpha=0.5, lw=2)
        ax1.add_patch(circle)

    # --- SUBPLOT 2: FEAT ---
    ax2.set_title("FEAT: Transformer Adaptation\n(Failure Mode: Prototype Collapse)", fontsize=16, pad=15)
    plot_support_images(ax2)

    for i in range(len(CLASSES)):
        orig_p = proto_net_prototypes[i]
        new_p = feat_prototypes[i]

        # Draw original prototype as a faint ghost
        ax2.scatter(orig_p[0], orig_p[1], color=COLORS[i], marker='*', s=300, alpha=0.2, zorder=3)
        
        # Draw adapted FEAT prototype
        ax2.scatter(new_p[0], new_p[1], color=COLORS[i], marker='*', s=800, edgecolor='black', linewidth=2, zorder=10)
        
        # Draw collapsed decision boundary overlapping others
        circle = patches.Circle((new_p[0], new_p[1]), radius=2.2, color=COLORS[i], fill=False, linestyle='-', alpha=0.3, lw=3)
        ax2.add_patch(circle)

        # Draw an arrow showing the massive shift caused by self-attention
        ax2.annotate('', xy=new_p, xytext=orig_p,
                     arrowprops=dict(arrowstyle='->, head_width=0.4, head_length=0.6', 
                                     color='black', lw=3, alpha=0.7))

    # --- FORMATTING ---
    for ax in [ax1, ax2]:
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=25, label='Class Prototype'),
        Line2D([0], [0], color='black', lw=3, label='Transformer Adaptation Shift')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    plt.savefig("prototype_comparison.png", dpi=300)

if __name__ == "__main__":
    create_image_prototype_comparison()