import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def create_7w10s_tradeoff_plot():
    # Extracted from 7-way 10-shot tasks in the provided logs
    # Format: "Algorithm": {"Backbone": (Latency_ms, Accuracy_percent)}
    data = {
        "ProtoNet": {
            "MobileNetV4": (0.83, 22.00),
            "EfficientNetV2-S": (1.53, 99.27),
            "Wide-ResNet50-2": (0.71, 98.62),
        },
        "FEAT": {
            "MobileNetV4": (0.88, 26.43),
            "EfficientNetV2-S": (1.47, 85.86),
            "Wide-ResNet50-2": (1.00, 20.64),
        },
        "RelationNet": {
            "MobileNetV4": (0.11, 96.41),
            "EfficientNetV2-S": (4.37, 99.05),
            "Wide-ResNet50-2": (1.60, 99.00),
        },
        "SiameseNet": {
            "MobileNetV4": (1.37, 85.28),
            "EfficientNetV2-S": (3.42, 85.72),
            "Wide-ResNet50-2": (1.42, 85.72),
        }
    }

    # Styling configuration
    algo_colors = {
        "ProtoNet": "#27ae60",     # Green
        "FEAT": "#e74c3c",         # Red
        "RelationNet": "#f39c12",  # Orange
        "SiameseNet": "#2980b9"    # Blue
    }
    
    backbone_markers = {
        "MobileNetV4": "o",        # Circle
        "EfficientNetV2-S": "s",   # Square
        "Wide-ResNet50-2": "^"     # Triangle
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the data points
    for algo, backbones in data.items():
        for backbone, (lat, acc) in backbones.items():
            ax.scatter(lat, acc, 
                       color=algo_colors[algo], 
                       marker=backbone_markers[backbone], 
                       s=250, # Marker size
                       edgecolor='black', 
                       linewidth=1.5,
                       alpha=0.9,
                       zorder=3)

    # --- HIGHLIGHT THE SWEET SPOT ---
    # Coordinates for ProtoNet + EfficientNetV2-S (Highest Accuracy)
    sweet_spot_x, sweet_spot_y = data["ProtoNet"]["EfficientNetV2-S"]
    
    # Draw a highlighted circle around it
    highlight = patches.Circle((sweet_spot_x, sweet_spot_y), radius=0.15, 
                               fill=False, edgecolor='red', linestyle='--', linewidth=2, zorder=1)
    ax.add_patch(highlight)
    
    # Add annotation pointing to the sweet spot
    ax.annotate('Optimal Edge Deployment\n(High Acc, Low Latency)', 
                xy=(sweet_spot_x, sweet_spot_y + 1), 
                xytext=(sweet_spot_x + 0.6, sweet_spot_y - 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                fontsize=12, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Optional: Highlight Wide-ResNet50-2 as the ultra-low latency alternative
    alt_x, alt_y = data["ProtoNet"]["Wide-ResNet50-2"]
    ax.annotate('Ultra-Low Latency\nAlternative', 
                xy=(alt_x, alt_y + 1), 
                xytext=(alt_x - 0.1, alt_y - 15),
                arrowprops=dict(arrowstyle="->", color='green', lw=2),
                fontsize=10, fontweight='bold', color='green', ha='center')


    # --- FORMATTING AND LABELS ---
    ax.set_title("Accuracy vs. Inference Latency Trade-off (7-Way 10-Shot Task)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Inference Latency per Query/Pair (ms) →\n(Lower is Better)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Task Accuracy (%) ↑\n(Higher is Better)", fontsize=14, fontweight='bold')
    
    # Adjusted axis limits to fit the wider range of accuracies (14% to 97%)
    ax.set_xlim(0.4, 4.0)
    ax.set_ylim(10, 105)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)

    # --- CUSTOM LEGENDS ---
    # 1. Algorithm Legend (Colors)
    algo_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           markersize=12, markeredgecolor='black', label=algo) 
                    for algo, color in algo_colors.items()]
    legend1 = ax.legend(handles=algo_handles, title="Meta-Learning Strategy", 
                        loc="lower right", fontsize=11, title_fontsize=12)
    ax.add_artist(legend1)

    # 2. Backbone Legend (Markers)
    backbone_handles = [Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', 
                               markersize=12, markeredgecolor='black', label=bb) 
                        for bb, marker in backbone_markers.items()]
    ax.legend(handles=backbone_handles, title="Feature Backbone", 
              loc="lower center", fontsize=11, title_fontsize=12)

    # Add Quadrant Shading for visual impact
    ax.axhspan(80, 105, xmin=0, xmax=0.35, facecolor='#2ecc71', alpha=0.1, zorder=0) # 0.35 roughly maps to ~1.6ms
    ax.text(0.5, 99, "High Performance Edge Zone", fontsize=12, color='#27ae60', fontstyle='italic', alpha=0.8)

    plt.tight_layout()
    # plt.savefig('accuracy_latency_tradeoff_7w10s.pdf', dpi=300) # Uncomment to save
    plt.show()
    plt.savefig("accuracy_latency_tradeoff_7w10s.png", dpi=300)

if __name__ == "__main__":
    create_7w10s_tradeoff_plot()