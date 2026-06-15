# Vẽ lại bar_repro_vs_paper từ CSV local, nhãn "Nhóm"/"Bài báo" (không cần Colab)
import csv, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

rows = list(
    csv.DictReader(
        open("4DGaussians/reports/comparison_repro_vs_paper.csv", encoding="utf-8")
    )
)
metrics = ["PSNR", "SSIM", "MS-SSIM", "LPIPS-vgg"]  # chỉ metric chất lượng
data = {m: [] for m in metrics}
for r in rows:
    if r["metric"] in metrics:
        data[r["metric"]].append((r["dataset"], float(r["mine"]), float(r["paper"])))

avail = [m for m in metrics if data[m]]
fig, axes = plt.subplots(1, len(avail), figsize=(4.5 * len(avail), 4))
if len(avail) == 1:
    axes = [axes]
for ax, m in zip(axes, avail):
    items = data[m]
    dss = [d for d, _, _ in items]
    mine = [mi for _, mi, _ in items]
    paper = [pa for _, _, pa in items]
    x = np.arange(len(dss))
    w = 0.38
    b1 = ax.bar(x - w / 2, mine, w, label="Nhóm", color="#1f77b4")
    b2 = ax.bar(x + w / 2, paper, w, label="Bài báo", color="#ff7f0e")
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_title(m)
    ax.set_xticks(x)
    ax.set_xticklabels(dss, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Trục 1 — Tái lập vs Công bố")
fig.tight_layout()
fig.savefig("report/images/bar_repro_vs_paper.png", bbox_inches="tight")
fig.savefig("report/images/bar_repro_vs_paper.pdf", bbox_inches="tight")
print("Da ve lai bar_repro_vs_paper (Nhom/Bai bao) ->", [d for d in data])
