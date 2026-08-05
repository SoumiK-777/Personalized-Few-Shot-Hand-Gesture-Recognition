import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_pipeline_diagram():
    # Increased figure width to prevent text merging and height for image placeholders
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Define the core blocks of your system
    nodes = [
        ("RGB Camera", "Driver Input\n(e.g., 'Fist')", "#ecf0f1", "black"),
        ("Preprocessing", "Hand Detection\n& Cropping", "#bdc3c7", "black"),
        ("EfficientNetV2-S", "Feature Extractor\n(Pretrained)", "#2980b9", "white"),
        ("ProtoNet Head", "Few-Shot Mapping\n(Distance to Prototype)", "#27ae60", "white"),
        ("Infotainment", "System Command\n(e.g., 'Open Map')", "#f39c12", "white")
    ]

    # Data flowing between the blocks
    flow_labels = [
        "Raw Frame", 
        "224x224 ROI", 
        "1280-D Vector", 
        "Class Match"
    ]

    # Layout dimensions
    box_w = 2.8   # Wider boxes
    box_h = 1.6   # Taller boxes
    gap = 1.4     # Larger gap between boxes for text
    y_center = 1.8 # Lowered the text boxes to make room for images
    img_y_center = 5.0 # Y-coordinate for the image placeholders

    for i, (title, subtext, bg_color, text_color) in enumerate(nodes):
        x = i * (box_w + gap)

        # 1. Draw the Main Text Block
        bbox = FancyBboxPatch((x, y_center - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.1,rounding_size=0.15",
                              ec="black", fc=bg_color, lw=1.5, zorder=3)
        ax.add_patch(bbox)

        # Main Block Title
        ax.text(x + box_w/2, y_center + 0.25, title, ha='center', va='center',
                fontsize=13, fontweight='bold', color=text_color, zorder=4)
        
        # Subtext explaining the block's function
        ax.text(x + box_w/2, y_center - 0.3, subtext, ha='center', va='center',
                fontsize=11, color=text_color, zorder=4, fontstyle='italic')

        # 2. Draw the Image Placeholder Block Above
        img_placeholder = patches.Rectangle((x + 0.2, img_y_center - 1.2), box_w - 0.4, 2.4, 
                                            fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
        ax.add_patch(img_placeholder)
        ax.text(x + box_w/2, img_y_center, "[ Insert Image ]", ha='center', va='center', 
                fontsize=10, color='gray', fontweight='bold')

        # 3. Draw arrows and data flow labels to the next block
        if i < len(nodes) - 1:
            next_x = (i + 1) * (box_w + gap)
            arrow_start_x = x + box_w
            arrow_end_x = next_x
            
            # Thick arrow connecting the text blocks
            ax.annotate('', xy=(arrow_end_x, y_center), xytext=(arrow_start_x, y_center),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.6",
                                        color="#2c3e50", lw=3),
                        zorder=2)
            
            # Data flow text hovering above the arrow (now with plenty of room)
            ax.text((arrow_start_x + arrow_end_x)/2, y_center + 0.2, flow_labels[i], 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#34495e')

            # Optional: Draw a subtle arrow connecting the image placeholders too
            ax.annotate('', xy=(arrow_end_x + 0.2, img_y_center), xytext=(arrow_start_x - 0.2, img_y_center),
                        arrowprops=dict(arrowstyle="->", color="lightgray", lw=2, linestyle='--'),
                        zorder=1)

    # Draw an encompassing dotted box to represent the "Edge Compute Node"
    # Wrapping around Preprocessing, Backbone, and ProtoNet (Nodes 1, 2, and 3)
    start_edge_x = 1 * (box_w + gap) - 0.3
    end_edge_x = 3 * (box_w + gap) + box_w + 0.3
    edge_width = end_edge_x - start_edge_x
    
    rect = patches.Rectangle((start_edge_x, 0.4), edge_width, 3.2, linewidth=2, edgecolor='gray', 
                             facecolor='none', linestyle='--', zorder=1)
    ax.add_patch(rect)
    ax.text(start_edge_x + 0.2, 0.6, "Deployed on Vehicle Compute Node", 
            fontsize=12, color='gray', fontstyle='italic')

    plt.tight_layout()
    plt.show()
    plt.savefig("system_pipeline_diagram.png", dpi=300)

if __name__ == "__main__":
    create_pipeline_diagram()