"""Bảng màu và helper vẽ biểu đồ dùng chung cho các hình trong báo cáo."""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # thứ tự cố định: blue, orange, aqua, yellow, magenta
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + BLUE_RAMP)

LABELS_VI = ["Không hài lòng", "Trung tính", "Hài lòng"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def plot_confusion(cm, title, path, normalize=True):
    """Ma trận nhầm lẫn: màu theo tỉ lệ theo hàng (recall), chữ ghi số lượng và %."""
    rows = cm.sum(axis=1, keepdims=True)
    share = cm / rows.clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(share if normalize else cm, cmap=SEQ_CMAP, vmin=0, vmax=1 if normalize else None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dark = share[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.1%}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if dark else TEXT)
    ax.set_xticks(range(3), LABELS_VI, fontsize=8.5)
    ax.set_yticks(range(3), LABELS_VI, fontsize=8.5)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(path)
    plt.close(fig)


def grouped_bars(groups, series, values, title, ylabel, path, ylim=None, fmt="{:.3f}", labels=True, figwidth=None):
    """Biểu đồ cột nhóm. values[s][g] là giá trị của series s tại nhóm g (NaN = không có cột)."""
    import numpy as np

    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(figwidth or max(5.5, 1.5 * len(groups) + 2), 3.6))
    ax.grid(axis="y", zorder=0)
    for k, s in enumerate(series):
        pos = x - 0.4 + width * (k + 0.5)
        bars = ax.bar(pos, values[k], width * 0.92, color=SERIES[k], label=s, zorder=2)
        for b, v in zip(bars, values[k]):
            if not labels or v != v:  # v != v: NaN
                continue
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_2)
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, color=TEXT, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    ax.tick_params(axis="x", length=0)
    fig.savefig(path)
    plt.close(fig)


def confusion_grid(cms, titles, path, ncols=3):
    """Nhiều ma trận nhầm lẫn trong một hình (cùng thang màu theo tỉ lệ hàng)."""
    import numpy as np

    nrows = int(np.ceil(len(cms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.7 * nrows))
    for ax, cm, title in zip(np.ravel(axes), cms, titles):
        cm = np.asarray(cm)
        share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if share[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(3), ["KHL", "TT", "HL"])
        ax.set_yticks(range(3), ["KHL", "TT", "HL"])
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in np.ravel(axes)[len(cms):]:
        ax.axis("off")
    fig.supxlabel("Nhãn dự đoán  (KHL = Không hài lòng, TT = Trung tính, HL = Hài lòng)", fontsize=9, color=TEXT_2)
    fig.supylabel("Nhãn thực tế", fontsize=9, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ci_dotplot(names, values, lows, highs, title, subtitle, path, ref=None, ref_label=None, highlight=0):
    """Macro-F1 kèm khoảng tin cậy cho từng mô hình (một hàng / mô hình), giá trị ghi ở cột phải."""
    import numpy as np

    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(names) + 1.4))
    ax.grid(axis="x", zorder=0)
    for k, (yy, v, lo, hi) in enumerate(zip(y, values, lows, highs)):
        color = SERIES[0] if k == highlight else "#c9c8c0"
        ax.plot([lo, hi], [yy, yy], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([v], [yy], s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{v:.3f} [{lo:.3f}–{hi:.3f}]".replace(".", ","), (1.02, yy), xycoords=("axes fraction", "data"),
                    va="center", fontsize=9, color=TEXT if k == highlight else TEXT_2,
                    fontweight="bold" if k == highlight else "normal")
    if ref is not None:
        ax.axvline(ref, color=TEXT_2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.annotate(ref_label, (ref, y.max() + 0.6), ha="right", va="bottom", fontsize=8.5, color=TEXT_2)
    ax.set_yticks(y, names)
    for t, k in zip(ax.get_yticklabels(), range(len(names))):
        t.set_fontweight("bold" if k == highlight else "normal")
    ax.set_ylim(-0.6, y.max() + 1.1)
    ax.set_xlabel("Macro-F1")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{title}\n", loc="left", fontsize=11.5, color=TEXT, fontweight="bold")
    ax.text(0, 1.0, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2, va="bottom")
    fig.savefig(path)
    plt.close(fig)
