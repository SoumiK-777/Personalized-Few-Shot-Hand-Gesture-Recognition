import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# --- CONFIGURATION ---
BASE_PATH = "cropped_images/Indian/train/"
WAYS = 5  
SHOTS = 5 
QUERIES = 2 
DEMOGRAPHIC = "Indian"

ALLOWED_GESTURES = [
    "four", "like", "one", "peace_inverted", "stop", "three", "three_gun",
    "dislike", "grabbing", "little_finger", "palm", "stop_inverted", "three2",
    "thumb_index", "two_up", "fist", "grip", "middle_finger", "ok", "peace",
    "rock", "three3", "thumb_index2", "two_up_inverted"
]

COLORS = ['#2c3e50', '#8e44ad', '#2980b9', '#27ae60', '#d35400']

def get_random_image(class_name):
    """Fetches a random image path from the given class folder."""
    folder_path = os.path.join(BASE_PATH, class_name)
    if not os.path.exists(folder_path):
        return None
    
    # Get all valid image files
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        return None
        
    return os.path.join(folder_path, random.choice(images))

def draw_image(ax, img_path, border_color, is_query=False):
    """Draws an image on the axis with a colored border."""
    if img_path and os.path.exists(img_path):
        img = plt.imread(img_path)
        ax.imshow(img)
    else:
        # Fallback if image is missing/not found
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='#ecf0f1', edgecolor='none'))
        ax.text(0.5, 0.5, "Image\nMissing", ha='center', va='center', fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Add border. Query images can have a distinct look, but let's keep borders thick for both
    # to show the class assignment clearly.
    linewidth = 5 if is_query else 3
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=linewidth, edgecolor=border_color, 
                             facecolor='none', transform=ax.transAxes)
    ax.add_patch(rect)
    ax.axis('off')

def create_episodic_diagram():
    # 1. Sample the episode
    sampled_classes = random.sample(ALLOWED_GESTURES, WAYS)
    display_classes = [c.replace('_', ' ').title() for c in sampled_classes]

    # 2. Setup Figure with adjusted margins to prevent overlapping!
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.05, right=0.95) # Gives top 20% to titles
    
    fig.suptitle(f'Episodic Formulation: {WAYS}-Way {SHOTS}-Shot Task\nDemographic Homogeneity: "{DEMOGRAPHIC}"', 
                 fontsize=20, fontweight='bold', y=0.95)

    main_gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.15)

    # --- SUPPORT SET ---
    support_gs = gridspec.GridSpecFromSubplotSpec(SHOTS + 1, WAYS, subplot_spec=main_gs[0], hspace=0.15, wspace=0.1)
    
    ax_sup_title = fig.add_subplot(main_gs[0])
    ax_sup_title.axis('off')
    ax_sup_title.set_title("Support Set (Labeled Data)", fontsize=16, fontweight='bold', pad=10)

    for w in range(WAYS):
        # Column Headers
        ax_header = fig.add_subplot(support_gs[0, w])
        ax_header.axis('off')
        ax_header.text(0.5, 0.2, f'{display_classes[w]}', 
                       fontsize=13, ha='center', va='center', fontweight='bold', color=COLORS[w])

        # Shots (Images)
        for s in range(SHOTS):
            ax = fig.add_subplot(support_gs[s + 1, w])
            img_path = get_random_image(sampled_classes[w])
            draw_image(ax, img_path, COLORS[w])
            
            # Add Y-axis labels for the first column only to keep it clean
            if w == 0:
                ax.text(-0.15, 0.5, f"Shot {s+1}", transform=ax.transAxes, 
                        ha='right', va='center', fontsize=11, fontweight='bold')

    # --- QUERY SET ---
    # Made width slightly smaller to prevent query text overlap
    query_gs = gridspec.GridSpecFromSubplotSpec(QUERIES + 1, WAYS, subplot_spec=main_gs[1], hspace=0.15, wspace=0.1)
    
    ax_query_title = fig.add_subplot(main_gs[1])
    ax_query_title.axis('off')
    ax_query_title.set_title("Query Set (Unseen)", fontsize=16, fontweight='bold', pad=10)

    for w in range(WAYS):
        # Column headers (Simplified to prevent text overlap)
        ax_header = fig.add_subplot(query_gs[0, w])
        ax_header.axis('off')
        ax_header.text(0.5, 0.2, f'?', fontsize=18, ha='center', va='center', color='gray', fontweight='bold')

        # Queries (Images)
        for q in range(QUERIES):
            ax = fig.add_subplot(query_gs[q + 1, w])
            img_path = get_random_image(sampled_classes[w])
            draw_image(ax, img_path, COLORS[w], is_query=True)

            if w == 0:
                ax.text(-0.15, 0.5, f"Img {q+1}", transform=ax.transAxes, 
                        ha='right', va='center', fontsize=11, fontweight='bold', color='gray')

    # --- BOUNDING BOX ---
    rect = patches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=2, edgecolor='black', 
                             facecolor='none', transform=fig.transFigure, linestyle='--')
    fig.patches.append(rect)
    fig.text(0.03, 0.04, 'One Training Episode', fontsize=14, fontstyle='italic')

    plt.savefig('episodic_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_episodic_diagram()