import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


rows = [
    ("Heterogeneous (ours)", "CGCNN", 0.195, 0.006),
    ("Heterogeneous (ours)", "MEGNet", 0.206, 0.006),
    ("Sparse", "CGCNN", 0.599, 0.399),
    ("Sparse", "MEGNet", 0.358, 0.005),
    ("Homogeneous", "CGCNN", 0.210, 0.005),
    ("Homogeneous", "MEGNet", 0.209, 0.005),
    ("Homogeneous", "ALIGNN", 0.198, 0.002),
    ("Attention", "CGCNN", 0.204, 0.003),
    ("Attention", "MEGNet", 0.225, 0.006),
    ("Attention", "DefiNet", 0.208, 0.005),
    ("Descriptors", "CatBoost", 0.250, 0.004),
]

colors_by_type = {
    "Heterogeneous (ours)": "#2f7f73",
    "Sparse": "#c96f4a",
    "Homogeneous": "#6e8fc7",
    "Attention": "#9a7cc4",
    "Descriptors": "#b99a45",
}

values = np.array([row[2] for row in rows])
errors = np.array([row[3] for row in rows])
labels = [row[1] for row in rows]
colors = [colors_by_type[row[0]] for row in rows]
x = np.arange(len(rows))

plt.rcParams.update({
    "font.family": "Arial",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(11.2, 5.6))
bars = ax.bar(
    x,
    values,
    color=colors,
    edgecolor="black",
    linewidth=0.55,
)

ax.set_ylabel("Adsorption Energy MAE (eV)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=10)
ax.set_ylim(0, 0.78)
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
ax.set_axisbelow(True)

for bar, value, error in zip(bars, values, errors):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.014,
        f"{value:.3f} ± {error:.3f}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        rotation=0,
    )

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5)
    for color in colors_by_type.values()
]
ax.legend(
    legend_handles,
    list(colors_by_type.keys()),
    ncol=5,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    fontsize=9,
    columnspacing=1.2,
)

fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.18)

output = Path(__file__).with_name("oc20_rs2re_adsorption_mae_bar_no_error_labeled.png")
fig.savefig(output, dpi=300)
plt.close(fig)
print(output)
