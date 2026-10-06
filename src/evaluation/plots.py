from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.agent.schemas import Alarm


def plot_alarm_policy_comparison(
    score: pd.Series,
    threshold: float,
    original: list[Alarm],
    retained: list[Alarm],
    events: list[dict],
    output_path: Path,
) -> None:
    new_events = [event for event in events if event["status"] != "excluded_carry_in"]
    retained_ids = {alarm.id for alarm in retained}
    suppressed = [alarm for alarm in original if alarm.id not in retained_ids]
    panels = [("Held-out period", score.index[0], score.index[-1])]
    for event in new_events:
        start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
        panels.append((f'{event["pipe"]}: new leak starting {start:%Y-%m-%d %H:%M}', max(score.index[0], start - pd.Timedelta(days=1)), min(score.index[-1], end + pd.Timedelta(days=1))))
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 4 * len(panels)), squeeze=False)
    for axis, (title, start, end) in zip(axes[:, 0], panels):
        visible = score.loc[start:end]
        axis.plot(visible.index, visible, color="#286090", linewidth=0.9, label="Anomaly score")
        axis.axhline(threshold, color="#303030", linestyle="--", linewidth=1.2, label="Unchanged threshold")
        first = True
        for event in new_events:
            onset, repair = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
            if onset <= end and repair >= start:
                axis.axvspan(max(onset, start), min(repair, end), color="#db737b", alpha=0.15, label="New leak active (ground truth)" if first else None)
                first = False
        for alarms, marker, color, label in (
            (suppressed, "x", "#c1444f", "Suppressed notifications"),
            (retained, "^", "#167747", "Retained notifications"),
        ):
            selected = [alarm for alarm in alarms if start <= pd.Timestamp(alarm.timestamp) <= end]
            axis.scatter([alarm.timestamp for alarm in selected], [alarm.score for alarm in selected], marker=marker, color=color, s=65, linewidths=1.7, label=label, zorder=4)
        axis.set_title(title, fontsize=12, loc="left")
        axis.set_ylabel("Anomaly score")
        axis.set_xlim(start, end)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
        axis.tick_params(axis="x", labelrotation=20)
    axes[-1, 0].set_xlabel("Dataset timestamp")
    fig.suptitle(f"Alarm de-duplication: {len(original)} notifications to {len(retained)}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
