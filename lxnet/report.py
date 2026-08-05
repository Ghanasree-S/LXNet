"""Turn results.json into the tables and figures that go in docs/RESULTS.md.

The headline this report exists to deliver is the gap between two protocols run
on identical data: a random split, where 40% of test images have a near-duplicate
twin in training, and a group-aware split where none do.

Every figure is rendered twice, once per theme, and referenced from a <picture>
element so the plots follow the reader's GitHub theme instead of burning a white
rectangle into a dark page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Theme:
    """One column of the validated palette. See the dataviz palette reference."""

    name: str
    surface: str
    ink: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    ramp: tuple[str, ...]

    @property
    def honest(self) -> str:
        return self.series[0]

    @property
    def leaky(self) -> str:
        return self.series[1]


# Slots are assigned in fixed order and never cycled. Both columns pass the
# validator's five checks against their own surface (light #fcfcfb, dark #1a1a19).
LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    muted="#52514e",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    ramp=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    muted="#c3c2b7",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
    # Sequential recedes toward the surface, so on a dark page near-zero is dark.
    ramp=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)
THEMES = (LIGHT, DARK)

# explain.py renders its panel against the light surface.
SURFACE, INK, INK_MUTED = LIGHT.surface, LIGHT.ink, LIGHT.muted
GRID = LIGHT.grid
SERIES = list(LIGHT.series)
BLUE_RAMP = list(LIGHT.ramp)
HONEST, LEAKY = LIGHT.honest, LIGHT.leaky


def _style(ax, t: Theme, yaxis_grid: bool = True):
    """Recessive axes: the data is the ink, the frame is not."""
    ax.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
    ax.tick_params(colors=t.muted, labelsize=9)
    if yaxis_grid:
        ax.grid(axis="y", color=t.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _fig(t: Theme, width=7.5, height=4.2, yaxis_grid: bool = True):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(t.surface)
    return fig, _style(ax, t, yaxis_grid)


def _save(fig, out: Path, t: Theme) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=t.surface)
    plt.close(fig)
    return out


def _themed(stem: Path, t: Theme) -> Path:
    """docs/fig_x.png for light, docs/fig_x-dark.png for dark."""
    return stem if t.name == "light" else stem.with_name(f"{stem.stem}-dark{stem.suffix}")


def build_comparison_table(grouped: dict, random_: dict | None) -> list[dict]:
    """One row per model: honest accuracy, leaky accuracy, and the inflation between."""
    rows = []
    random_holdout = (random_ or {}).get("holdout", {})
    for name, metrics in grouped.get("holdout", {}).items():
        leaky = random_holdout.get(name)
        g_acc = metrics.get("accuracy")
        r_acc = leaky.get("accuracy") if leaky else None
        rows.append(
            {
                "model": name,
                "params": metrics.get("params"),
                "grouped_accuracy": g_acc,
                "grouped_f1": metrics.get("f1_macro"),
                "random_accuracy": r_acc,
                "inflation": (r_acc - g_acc) if (r_acc is not None and g_acc is not None) else None,
            }
        )
    return sorted(rows, key=lambda r: r["grouped_accuracy"] or 0, reverse=True)


def markdown_table(rows: list[dict]) -> str:
    """Render the comparison as a GitHub table. Missing arms render as an em dash."""

    def pct(v):
        return "—" if v is None else f"{100 * v:.1f}"

    def params(v):
        return "—" if v is None else (f"{v / 1e6:.2f} M" if v >= 1e6 else f"{v / 1e3:.0f} K")

    def f1(v):
        return "—" if v is None else f"{v:.3f}"

    def inflation(v):
        return "—" if v is None else f"+{100 * v:.1f}"

    lines = [
        "| Model | Params | Group-aware acc. (%) | Group-aware F1 "
        "| Random-split acc. (%) | Inflation (pp) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {params(r['params'])} | {pct(r['grouped_accuracy'])} "
            f"| {f1(r['grouped_f1'])} | {pct(r['random_accuracy'])} | {inflation(r['inflation'])} |"
        )
    return "\n".join(lines) + "\n"


def summarise(grouped: dict, random_: dict | None, leaked_fraction: float) -> str:
    """A short prose statement of the finding, with the numbers filled in."""
    rows = build_comparison_table(grouped, random_)
    lx = next((r for r in rows if r["model"] == "LXNet"), None)
    best = rows[0] if rows else None
    parts = [
        f"Under a random split {100 * leaked_fraction:.1f}% of test images have a "
        f"near-duplicate in the training set; under the group-aware split, none do."
    ]
    if lx and lx["grouped_accuracy"] is not None:
        parts.append(
            f"LXNet scores {100 * lx['grouped_accuracy']:.1f}% on the group-aware test set"
            + (
                f", against {100 * lx['random_accuracy']:.1f}% on the random split "
                f"(+{100 * lx['inflation']:.1f} pp of leakage)."
                if lx["random_accuracy"] is not None
                else "."
            )
        )
    if best and best["model"] != "LXNet":
        parts.append(
            f"The strongest model overall is {best['model']} at "
            f"{100 * best['grouped_accuracy']:.1f}%."
        )
    return " ".join(parts)


# --- figures -----------------------------------------------------------------


def _gap_dumbbell(rows: list[dict], stem: Path, t: Theme) -> Path | None:
    """The headline: how far each model falls once its twins are taken away."""
    paired = [r for r in rows if r["random_accuracy"] is not None]
    if not paired:
        return None
    paired = sorted(paired, key=lambda r: r["inflation"] or 0)

    fig, ax = _fig(t, 8.0, max(2.9, 0.85 * len(paired) + 2.0), yaxis_grid=False)
    ax.grid(axis="x", color=t.grid, linewidth=0.8)
    y = np.arange(len(paired))

    for yi, r in zip(y, paired, strict=True):
        g, rnd = 100 * r["grouped_accuracy"], 100 * r["random_accuracy"]
        ax.plot([g, rnd], [yi, yi], color=t.axis, linewidth=2.5, zorder=1, solid_capstyle="round")
        ax.scatter([g], [yi], s=150, color=t.honest, edgecolor=t.surface, linewidth=2, zorder=3)
        ax.scatter([rnd], [yi], s=150, color=t.leaky, edgecolor=t.surface, linewidth=2, zorder=3)
        ax.text(
            (g + rnd) / 2,
            yi + 0.30,
            f"+{rnd - g:.1f} pp",
            ha="center",
            fontsize=9,
            color=t.ink,
            fontweight="bold",
        )
        ax.text(g - 0.55, yi, f"{g:.1f}", ha="right", va="center", fontsize=9, color=t.muted)
        ax.text(rnd + 0.55, yi, f"{rnd:.1f}", ha="left", va="center", fontsize=9, color=t.muted)

    ax.set_yticks(y, [r["model"] for r in paired], color=t.ink, fontsize=10)
    lo = min(100 * r["grouped_accuracy"] for r in paired)
    hi = max(100 * r["random_accuracy"] for r in paired)
    ax.set_xlim(lo - 6, hi + 4)
    ax.set_ylim(-0.62, len(paired) - 0.38)
    ax.set_xlabel("Test accuracy (%)", color=t.muted, fontsize=9)
    ax.set_title(
        "What leakage was worth\nEach line is one model measured twice on identical data",
        color=t.ink,
        fontsize=12,
        loc="left",
        pad=14,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            color=t.honest,
            label="Group-aware split (no twin in training)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            color=t.leaky,
            label="Random split (paper protocol)",
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=t.muted, loc="lower right")
    return _save(fig, _themed(stem, t), t)


def _efficiency(rows: list[dict], stem: Path, t: Theme) -> Path | None:
    """Parameters against honest accuracy -- the paper's central claim, plotted.

    One series, direct-labelled: four models compared all-pairs would exceed the
    palette's three-slot cap for non-adjacent forms, and identity here is carried
    by the label, not the hue.
    """
    pts = [r for r in rows if r.get("params") and r["grouped_accuracy"] is not None]
    if len(pts) < 2:
        return None

    fig, ax = _fig(t, 7.6, 4.8)
    xs = [r["params"] for r in pts]
    ys = [100 * r["grouped_accuracy"] for r in pts]
    ax.scatter(xs, ys, s=170, color=t.honest, edgecolor=t.surface, linewidth=2, zorder=3)
    for r, x, y in zip(pts, xs, ys, strict=True):
        # Drop the label below when a neighbour sits just above it -- the
        # backbones cluster within ~1 pp and their labels would otherwise collide.
        crowded_above = any(
            other is not r
            and 0 < (100 * other["grouped_accuracy"] - y) < 2.0
            and 0.4 < other["params"] / x < 2.5
            for other in pts
        )
        # Keep the right-hand labels inside the axes.
        near_right = x > max(xs) / 1.6
        ax.annotate(
            f"{r['model']}\n{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(-13 if near_right else 13, -24 if crowded_above else 13),
            ha="right" if near_right else "left",
            fontsize=9,
            color=t.ink,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)", color=t.muted, fontsize=9)
    ax.set_ylabel("Group-aware test accuracy (%)", color=t.muted, fontsize=9)
    ax.set_xlim(min(xs) / 3.0, max(xs) * 3.0)
    ax.set_ylim(min(ys) - 2.2, max(ys) + 1.8)
    ax.set_title(
        "Does the small model keep up?\nHonest accuracy against parameter count",
        color=t.ink,
        fontsize=12,
        loc="left",
        pad=14,
    )
    return _save(fig, _themed(stem, t), t)


def _bar_models(rows: list[dict], stem: Path, t: Theme) -> Path:
    fig, ax = _fig(t, 7.5, 4.0, yaxis_grid=False)
    ax.grid(axis="x", color=t.grid, linewidth=0.8)
    labels = [r["model"] for r in rows]
    vals = [100 * (r["grouped_accuracy"] or 0) for r in rows]
    ax.barh(np.arange(len(rows)), vals, 0.62, color=t.honest, edgecolor=t.surface, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, color=t.ink)
    ax.set_yticks(np.arange(len(rows)), labels, color=t.ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test accuracy (%), group-aware split", color=t.muted, fontsize=9)
    ax.set_title("Honest accuracy by model", color=t.ink, fontsize=12, loc="left", pad=12)
    return _save(fig, _themed(stem, t), t)


def _cv_folds(grouped: dict, stem: Path, t: Theme) -> Path | None:
    cv = grouped.get("cross_validation") or {}
    if not cv:
        return None
    fig, ax = _fig(t, 6.6, 4.2)
    names = list(cv)
    for i, name in enumerate(names):
        accs = [100 * f["accuracy"] for f in cv[name]["folds"]]
        # Folds land within a few hundredths of each other, so plotting them on
        # one vertical would hide a point behind another. Spread them evenly.
        dodge = np.linspace(-0.13, 0.13, len(accs)) if len(accs) > 1 else np.zeros(1)
        ax.scatter(
            i + dodge,
            accs,
            s=90,
            color=t.series[i % len(t.series)],
            edgecolor=t.surface,
            linewidth=2,
            zorder=3,
        )
        m = float(np.mean(accs))
        ax.plot([i - 0.24, i + 0.24], [m, m], color=t.ink, linewidth=2.5, zorder=4)
        ax.text(
            i + 0.30,
            m,
            f"mean {m:.2f}%\nsd {np.std(accs):.2f}",
            va="center",
            fontsize=9,
            color=t.ink,
        )
    ax.set_xticks(range(len(names)), names, color=t.ink)
    ax.set_xlim(-0.5, len(names) + 0.25)
    ax.set_ylabel("Fold accuracy (%)", color=t.muted, fontsize=9)
    ax.set_title(
        "Cross-validation folds, group-aware\nEach dot is one held-out fold, scored once",
        color=t.ink,
        fontsize=12,
        loc="left",
        pad=14,
    )
    return _save(fig, _themed(stem, t), t)


def _per_class(grouped: dict, class_names: list[str], stem: Path, t: Theme) -> Path | None:
    """Per-class recall for the best model -- where the difficulty actually is."""
    holdout = grouped.get("holdout") or {}
    # LXNet is the model under replication -- the backbones sit near 100% per
    # class, which makes for a chart that says nothing.
    candidates = [n for n in ("LXNet", *holdout) if holdout.get(n, {}).get("confusion_matrix")]
    if not candidates:
        return None
    name = candidates[0]
    cm = np.asarray(holdout[name]["confusion_matrix"], dtype=float)
    recall = 100 * np.diag(cm) / np.clip(cm.sum(axis=1), 1, None)
    order = np.argsort(recall)
    labels = [class_names[i] for i in order]

    fig, ax = _fig(t, 7.8, 4.8, yaxis_grid=False)
    ax.grid(axis="x", color=t.grid, linewidth=0.8)
    ax.barh(
        np.arange(len(order)), recall[order], 0.62, color=t.honest, edgecolor=t.surface, linewidth=2
    )
    for i, v in enumerate(recall[order]):
        ax.text(v + 0.9, i, f"{v:.1f}%", va="center", fontsize=9, color=t.ink)
    ax.set_yticks(np.arange(len(order)), labels, color=t.ink, fontsize=9)
    ax.invert_yaxis()  # lowest recall on top, matching the title
    ax.set_xlim(0, 112)
    ax.set_xlabel("Recall (%)", color=t.muted, fontsize=9)
    ax.set_title(
        f"Hardest classes first — {name}, group-aware test set",
        color=t.ink,
        fontsize=12,
        loc="left",
        pad=12,
    )
    return _save(fig, _themed(stem, t), t)


def _training_curves(histories: dict[str, dict], stem: Path, t: Theme) -> Path | None:
    """Validation accuracy per epoch. Four series max, on the adjacent pairlist."""
    usable = {k: v for k, v in histories.items() if v.get("val_accuracy")}
    if not usable:
        return None
    fig, ax = _fig(t, 7.8, 4.4)
    ends = []
    for i, (name, hist) in enumerate(sorted(usable.items())):
        ys = [100 * v for v in hist["val_accuracy"]]
        xs = np.arange(1, len(ys) + 1)
        colour = t.series[i % len(t.series)]
        ax.plot(xs, ys, color=colour, linewidth=2, zorder=3, label=name)
        ax.scatter(
            [xs[-1]], [ys[-1]], s=45, color=colour, edgecolor=t.surface, linewidth=2, zorder=4
        )
        ends.append([float(xs[-1]), float(ys[-1]), name, colour])

    # The models converge within a few points of each other, so the end labels
    # would land on top of one another. Push them apart, keeping their order.
    ends.sort(key=lambda e: -e[1])
    gap = 3.4
    for prev, cur in zip(ends, ends[1:], strict=False):
        cur[1] = min(cur[1], prev[1] - gap)
    for x, y, name, _colour in ends:
        ax.text(x + 1.1, y, name, va="center", fontsize=9, color=t.ink)

    ax.set_xlabel("Epoch", color=t.muted, fontsize=9)
    ax.set_ylabel("Validation accuracy (%)", color=t.muted, fontsize=9)
    ax.set_xlim(1, max(len(h["val_accuracy"]) for h in usable.values()) + 9)
    ax.legend(frameon=False, fontsize=9, labelcolor=t.muted, loc="lower right", ncol=2)
    ax.set_title(
        "Training, group-aware split\nEarly stopping restores the best epoch",
        color=t.ink,
        fontsize=12,
        loc="left",
        pad=14,
    )
    return _save(fig, _themed(stem, t), t)


def confusion_figure(
    matrix: np.ndarray, class_names: list[str], out: Path, title: str, t: Theme = LIGHT
) -> Path:
    """Sequential single-hue heatmap; row-normalised so classes are comparable."""
    matrix = np.asarray(matrix, dtype=float)
    norm = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", list(t.ramp))
    fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=200)
    fig.patch.set_facecolor(t.surface)
    ax.set_facecolor(t.surface)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(
        range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8, color=t.muted
    )
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8, color=t.muted)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if norm[i, j] >= 0.01:
                # A strong cell sits at the far end of the ramp from the surface,
                # so the surface colour is always the readable label on it --
                # dark ink on light cells, light ink on dark ones, both themes.
                strong = norm[i, j] > 0.55
                colour = t.surface if strong else t.muted
                ax.text(
                    j,
                    i,
                    f"{100 * norm[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=colour,
                )
    ax.set_xlabel("Predicted", color=t.muted, fontsize=9)
    ax.set_ylabel("True", color=t.muted, fontsize=9)
    ax.set_title(title, color=t.ink, fontsize=12, loc="left", pad=12)
    bar = fig.colorbar(im, ax=ax, shrink=0.8)
    bar.set_label("% of true class", color=t.muted, fontsize=9)
    bar.ax.tick_params(colors=t.muted, labelsize=8)
    bar.outline.set_edgecolor(t.axis)
    return _save(fig, out, t)


def _load_histories(history_dir: Path | None) -> dict[str, dict]:
    if history_dir is None:
        return {}
    return {
        p.stem.replace("history_", ""): json.loads(p.read_text())
        for p in sorted(Path(history_dir).glob("history_*.json"))
    }


def write_figures(
    grouped: dict, random_: dict | None, out_dir: Path, history_dir: Path | None = None
) -> list[Path]:
    """Write every figure, in both themes. Returns the light-mode paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_comparison_table(grouped, random_)
    histories = _load_histories(history_dir)

    from .data import CLASS_LABELS

    names = list(CLASS_LABELS.values())
    written: list[Path] = []

    for t in THEMES:
        produced = [
            _gap_dumbbell(rows, out_dir / "fig_gap.png", t),
            _bar_models(rows, out_dir / "fig_accuracy_by_model.png", t),
            _efficiency(rows, out_dir / "fig_efficiency.png", t),
            _cv_folds(grouped, out_dir / "fig_cv_folds.png", t),
            _per_class(grouped, names, out_dir / "fig_per_class.png", t),
            _training_curves(histories, out_dir / "fig_training.png", t),
        ]
        for model, metrics in (grouped.get("holdout") or {}).items():
            cm = metrics.get("confusion_matrix")
            if cm:
                produced.append(
                    confusion_figure(
                        np.array(cm),
                        names[: len(cm)],
                        _themed(out_dir / f"fig_confusion_{model}.png", t),
                        f"{model} — group-aware test set (row %)",
                        t,
                    )
                )
        if t.name == "light":
            written = [p for p in produced if p is not None]
    return written


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def leakage_fraction(data_root: str | Path = "dataset", seed: int = 42) -> float:
    """Share of random-split test images having a near-duplicate in training."""
    from .data import dedupe, group_samples, index_dataset, stratified_split

    samples, _ = dedupe(index_dataset(data_root))
    group_of = {s.path: g for g, members in group_samples(samples).items() for s in members}
    parts = stratified_split(samples, (0.70, 0.15, 0.15), seed=seed)
    train_groups = {group_of[s.path] for s in parts["train"]}
    leaked = sum(1 for s in parts["test"] if group_of[s.path] in train_groups)
    return leaked / len(parts["test"])


def picture(stem: str, alt: str) -> str:
    """A GitHub <picture> that swaps the figure with the reader's theme."""
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{stem}-dark.png">\n'
        f'  <img alt="{alt}" src="{stem}.png">\n'
        "</picture>"
    )


def _hero(rows: list[dict], grouped: dict, leaked: float) -> list[str]:
    """Four numbers that carry the whole result, before any prose."""
    lx = next((r for r in rows if r["model"] == "LXNet"), None)
    best = rows[0] if rows else None
    gap = (
        f"+{100 * lx['inflation']:.1f} pp" if lx and lx.get("inflation") is not None else "pending"
    )
    spread = (
        f"{100 * (best['grouped_accuracy'] - lx['grouped_accuracy']):.1f} pp"
        if best and lx and best["model"] != "LXNet"
        else "—"
    )
    cell = '<td align="center"><h2>{}</h2>{}</td>'
    return [
        "<table>",
        "<tr>",
        cell.format(
            f"{100 * leaked:.1f}%",
            "test set with a<br>training twin<br><sub>random split</sub>",
        ),
        cell.format(
            gap,
            "accuracy LXNet gains<br>from that leakage<br><sub>same data, same model</sub>",
        ),
        cell.format(
            spread,
            "LXNet trails the best<br>backbone once it is gone<br><sub>group-aware hold-out</sub>",
        ),
        cell.format(
            f"{grouped['dedupe']['kept']:,}",
            "byte-unique files<br>&rarr; 4,635 distinct X-rays<br><sub>30% redundancy</sub>",
        ),
        "</tr>",
        "</table>",
    ]


PROTOCOL_DIAGRAM = """```mermaid
flowchart TD
    A["6,743 image files"] --> B["drop 84 byte-identical<br/>+ 1 cross-class conflict"]
    B --> C["6,657 byte-unique files"]
    C --> D["perceptual hash<br/>(dhash, exact match)"]
    D --> E["4,635 visually distinct X-rays<br/>30% of the corpus is re-saved copies"]
    E --> F{"how do we split?"}
    F -->|"per image<br/>(paper protocol)"| G["Random split"]
    F -->|"per whole group"| H["Group-aware split"]
    G --> I["40.4% of test images<br/>have a twin in training"]
    H --> J["0% - every copy of an<br/>X-ray lands in one split"]
    I --> K["scored on pictures<br/>it already memorised"]
    J --> L["scored on genuinely<br/>unseen images"]

    style E fill:#eda100,stroke:#0b0b0b,color:#0b0b0b
    style I fill:#eb6834,stroke:#0b0b0b,color:#0b0b0b
    style J fill:#1baf7a,stroke:#0b0b0b,color:#0b0b0b
    style K fill:#eb6834,stroke:#0b0b0b,color:#0b0b0b
    style L fill:#1baf7a,stroke:#0b0b0b,color:#0b0b0b
```"""


def _cv_section(grouped: dict) -> list[str]:
    cv = grouped.get("cross_validation") or {}
    if not cv:
        return []
    out = ["## Cross-validation", "", picture("fig_cv_folds", "Per-fold accuracy"), ""]
    out += ["| Model | Folds | Accuracy (%) | Macro-F1 |", "|---|---|---|---|"]
    for name, block in cv.items():
        s = block["summary"]
        f1 = s.get("f1_macro_mean")
        out.append(
            f"| {name} | {len(block['folds'])} | "
            f"{100 * s['accuracy_mean']:.2f} ± {100 * s['accuracy_std']:.2f} | "
            + ("—" if f1 is None else f"{f1:.4f}")
            + " |"
        )
    out += [
        "",
        "Each fold's early-stopping monitor is carved out of that fold's *training* "
        "rows, so the reported fold is touched exactly once, at scoring time.",
        "",
    ]
    return out


def per_class_metrics(matrix) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support per class, derived from a confusion matrix."""
    cm = np.asarray(matrix, dtype=float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    hits = np.diag(cm)
    recall = hits / np.clip(support, 1, None)
    precision = hits / np.clip(predicted, 1, None)
    denom = np.clip(precision + recall, 1e-12, None)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / denom,
        "support": support,
    }


def _metrics_table(grouped: dict) -> list[str]:
    """Every headline metric for every model, not just accuracy."""
    holdout = grouped.get("holdout") or {}
    if not holdout:
        return []
    ordered = sorted(holdout.items(), key=lambda kv: kv[1].get("accuracy") or 0, reverse=True)
    lines = [
        "### Every metric, group-aware hold-out",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for name, m in ordered:
        lines.append(
            f"| {name} | {100 * m['accuracy']:.2f} | {100 * m['precision_macro']:.2f} "
            f"| {100 * m['recall_macro']:.2f} | {m['f1_macro']:.4f} |"
        )
    lines += [
        "",
        "Macro averaging throughout. The dataset runs 1,340 Normal against 544 Chest "
        "Changes, so a micro average would flatter any model that neglects the small "
        "classes.",
        "",
    ]
    return lines


def _per_class_tables(grouped: dict, class_names: list[str]) -> list[str]:
    """Per-class detail for LXNet, then recall for every model side by side."""
    holdout = grouped.get("holdout") or {}
    with_cm = {k: v for k, v in holdout.items() if v.get("confusion_matrix")}
    if not with_cm:
        return []

    focus = "LXNet" if "LXNet" in with_cm else next(iter(with_cm))
    stats = per_class_metrics(with_cm[focus]["confusion_matrix"])
    lines = [
        f"### Per-class detail — {focus}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for i, name in enumerate(class_names[: len(stats["support"])]):
        lines.append(
            f"| {name} | {100 * stats['precision'][i]:.1f} | {100 * stats['recall'][i]:.1f} "
            f"| {stats['f1'][i]:.3f} | {int(stats['support'][i])} |"
        )

    lines += ["", "### Recall by class, every model", "", "| Class |"]
    ordered = sorted(with_cm, key=lambda k: with_cm[k]["accuracy"], reverse=True)
    lines[-1] += "".join(f" {n} |" for n in ordered)
    lines.append("|---|" + "---|" * len(ordered))
    n_classes = len(next(iter(with_cm.values()))["confusion_matrix"])
    for i, name in enumerate(class_names[:n_classes]):
        row = f"| {name} |"
        for model in ordered:
            r = per_class_metrics(with_cm[model]["confusion_matrix"])["recall"][i]
            row += f" {100 * r:.1f} |"
        lines.append(row)
    lines += [
        "",
        "The weakest cells are the ones that matter clinically: every model gives up "
        "most of its accuracy on Pneumonia and Normal, the two classes a screening "
        "tool exists to separate.",
        "",
    ]
    return lines


def _cost_table(grouped: dict) -> list[str]:
    holdout = grouped.get("holdout") or {}
    if not holdout:
        return []
    lines = [
        "## Training cost",
        "",
        "| Model | Parameters | Epochs run | Wall clock | Accuracy per M params |",
        "|---|---|---|---|---|",
    ]
    for name, m in sorted(holdout.items(), key=lambda kv: kv[1].get("params") or 0):
        secs = m.get("train_seconds") or 0
        per_m = (100 * m["accuracy"]) / max(m["params"] / 1e6, 1e-9)
        lines.append(
            f"| {name} | {m['params']:,} | {m.get('epochs_run', '—')} | "
            f"{int(secs // 60)}m {int(secs % 60)}s | {per_m:,.1f} |"
        )
    lines += [
        "",
        "The last column is deliberately crude, but it is the paper's argument in one "
        "number: LXNet extracts far more accuracy per parameter than any backbone, "
        "while still finishing below all of them in absolute terms.",
        "",
    ]
    return lines


def _wilcoxon_section(grouped: dict) -> list[str]:
    w = grouped.get("wilcoxon") or {}
    cv = grouped.get("cross_validation") or {}
    if not w:
        return [
            "## Significance testing",
            "",
            f"Not run. The Wilcoxon signed-rank test compares two models across paired "
            f"folds, and only {len(cv)} model "
            f"({', '.join(cv) or 'none'}) was cross-validated — the heavy backbones are "
            "hold-out only for time reasons. Cross-validate a second model to populate "
            "this section.",
            "",
        ]
    lines = [
        "## Significance testing",
        "",
        "| Comparison | p | Median difference | Significant |",
        "|---|---|---|---|",
    ]
    for pair, res in w.items():
        lines.append(
            f"| {pair} | {res['p_value']:.4f} | {100 * res['median_difference']:+.2f} pp "
            f"| {'yes' if res['significant'] else 'no'} |"
        )
    lines.append("")
    return lines


def _repro_section(grouped: dict, random_: dict | None) -> list[str]:
    split = grouped["split"]
    lines = [
        "## Reproducibility record",
        "",
        "| | Group-aware arm | Random arm |",
        "|---|---|---|",
        f"| Split mode | `{grouped.get('split_mode')}` | "
        f"`{(random_ or {}).get('split_mode', '—')}` |",
        f"| Train / val / test | {split['train']} / {split['val']} / {split['test']} | "
        + (
            f"{random_['split']['train']} / {random_['split']['val']} / "
            f"{random_['split']['test']} |"
            if random_
            else "— |"
        ),
        "| Seed | 42 | 42 |",
        "| Max epochs | 40, early stopping patience 8 | 40, early stopping patience 8 |",
        "| Batch size | 32 | 32 |",
        "",
        "Both arms consume the same cached CLAHE'd tensor, so preprocessing is bit-for-bit "
        "identical between them. The only variable is the split function.",
        "",
        "Every figure on this page is generated by `python -m lxnet.report` from "
        "`results.json`; none is drawn by hand.",
        "",
    ]
    return lines


def build_results_md(
    grouped: dict, random_: dict | None, leaked: float, figures: list[Path]
) -> str:
    """The whole document. Sections appear only when their figure exists."""
    from .data import CLASS_LABELS

    class_names = list(CLASS_LABELS.values())
    rows = build_comparison_table(grouped, random_)
    have = {p.stem for p in figures}
    body: list[str] = [
        "# Results",
        "",
        "> A replication of Humayan et al. (PLOS One, 2026) that reports the number "
        "the paper's protocol hides.",
        "",
    ]
    body += _hero(rows, grouped, leaked)
    body += ["", "---", "", "## The finding", "", summarise(grouped, random_, leaked), ""]

    if "fig_gap" in have:
        body += [picture("fig_gap", "Accuracy under each protocol, per model"), ""]

    body += [
        "Both arms run on identical images, identical preprocessing and identical "
        "architectures. The only thing that changes is whether a re-saved copy of a "
        "training X-ray is allowed to appear in the test set.",
        "",
        "## How the two protocols differ",
        "",
        PROTOCOL_DIAGRAM,
        "",
        "Grouping uses exact perceptual-hash equality. Chest X-rays are near-identical "
        "by construction, so any Hamming tolerance chains distinct images together "
        "transitively — at threshold 3 the largest group reaches 1,680 images spanning "
        "8 of 9 classes. Exact equality under-counts mild recompression but never "
        "merges two different patients.",
        "",
        "## Hold-out comparison",
        "",
        markdown_table(rows),
    ]
    body += _metrics_table(grouped)

    if "fig_efficiency" in have:
        body += [
            "### Does the small model keep up?",
            "",
            "The paper's central claim is that ~0.35 M parameters match backbones "
            "50–70× larger. That claim is only testable on a split where the models "
            "can actually be told apart.",
            "",
            picture("fig_efficiency", "Parameters against group-aware accuracy"),
            "",
        ]
    if "fig_accuracy_by_model" in have:
        body += [picture("fig_accuracy_by_model", "Honest accuracy by model"), ""]

    body += _cv_section(grouped)

    body += ["## Where the errors are", ""]
    if "fig_per_class" in have:
        body += [picture("fig_per_class", "Per-class recall"), ""]
    body += _per_class_tables(grouped, class_names)

    body += ["### Confusion matrices", ""]
    for stem in sorted(p.stem for p in figures if p.stem.startswith("fig_confusion_")):
        model = stem.replace("fig_confusion_", "")
        body += [f"**{model}**", "", picture(stem, f"{model} confusion matrix"), ""]

    if "fig_training" in have:
        body += ["## Training", "", picture("fig_training", "Validation accuracy per epoch"), ""]
    body += _cost_table(grouped)
    body += _wilcoxon_section(grouped)

    body += [
        "## Dataset after cleaning",
        "",
        # Deliberately not summed into an "images indexed" total: the dedupe report
        # counts conflict *groups*, not the files they dropped, so the arithmetic
        # would be short by the conflicting copies.
        f"- byte-unique images retained: {grouped['dedupe']['kept']:,}",
        f"- byte-identical duplicates removed: {grouped['dedupe']['exact_duplicates']}",
        f"- cross-class label conflict groups removed: "
        f"{grouped['dedupe']['cross_class_conflicts']}",
        f"- split (train/val/test): {grouped['split']['train']}/"
        f"{grouped['split']['val']}/{grouped['split']['test']}",
        f"- random-split test images with a training twin: {100 * leaked:.1f}%",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python -m lxnet.train --out-dir runs/grouped --split-mode grouped --cv-models LXNet",
        "python -m lxnet.train --out-dir runs/random --split-mode random "
        "--models LXNet --cv-models LXNet",
        "python -m lxnet.report",
        "```",
        "",
    ]
    body += _repro_section(grouped, random_)
    body += [
        "<sub>Figures render light or dark to match your GitHub theme. Palette validated "
        "for colour-vision deficiency in both modes.</sub>",
    ]
    return "\n".join(body) + "\n"


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build figures and docs/RESULTS.md.")
    parser.add_argument("--grouped", type=Path, default=Path("runs/grouped/results.json"))
    parser.add_argument("--random", type=Path, default=Path("runs/random/results.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    args = parser.parse_args(argv)

    grouped = load(args.grouped)
    random_ = load(args.random) if args.random.exists() else None
    if random_ is None:
        print(f"note: {args.random} absent -- the leakage arm will read as pending")

    figures = write_figures(grouped, random_, args.out_dir, history_dir=args.grouped.parent)
    leaked = leakage_fraction(args.data_root)
    (args.out_dir / "RESULTS.md").write_text(
        build_results_md(grouped, random_, leaked, figures), encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'RESULTS.md'} and {2 * len(figures)} figure files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
