import argparse
import hashlib
import html
import importlib
import json
import math
import subprocess
import textwrap
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, TypedDict, cast

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from autojev.types import JSONValue

type Event = dict[str, JSONValue]
type Panel = Literal["full", "quick"]
type Cohort = Literal["public", "guard"]
GREEN = "#286b4d"
GRAY = "#87938f"
INK = "#273431"
PALE = "#c7dccb"
RAW = "#597d99"
MARGIN = 0.010
RUN_EVENTS = {
    "training_started",
    "training_step",
    "evaluation",
    "quick_evaluation",
    "training_paused",
    "training_finished",
    "training_failed",
    "run_restarted",
}
QUICK_NOTE = "Quick diagnostic: 1,000 public / 200 guard / 700 temperature-fit rows · never selects checkpoints"


class Series(TypedDict):
    run: str
    panel: Panel
    cohort: Cohort
    events: list[Event]
    steps: list[Event]
    evaluations: list[Event]
    reference: dict[str, float]
    total_steps: int


class HistoryRow(TypedDict):
    run: str
    step: int
    status: str
    detail: str
    recovery: str


def number(value: JSONValue | object, default: float = 0.0) -> float:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        else default
    )


def timestamp(event: Event) -> float:
    value = event.get("timestamp")
    if not isinstance(value, str):
        raise ValueError("Every report event needs a UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Report timestamps must have a time zone")
    return parsed.timestamp()


def load_events(path: Path) -> list[Event]:
    return parse_events(path.read_text())


def parse_events(content: str) -> list[Event]:
    lines = content.splitlines()

    if content and not content.endswith("\n"):
        lines = lines[:-1]
    events: list[Event] = []
    for line in lines:
        if line.strip():
            event = cast(Event, json.loads(line))
            if not isinstance(event.get("kind"), str):
                raise ValueError("Every report event needs a kind")
            timestamp(event)
            events.append(event)
    if not events:
        raise ValueError("No complete recorded events are available")
    return sorted(events, key=timestamp)


def metric_values(event: Event, key: str) -> dict[str, float]:
    value = event.get(key)
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for name in ("accuracy", "ece", "brier", "nll"):
        if name in value:
            metric = value[name]
            if not isinstance(metric, int | float) or isinstance(metric, bool) or not math.isfinite(metric):
                raise ValueError(f"Invalid recorded {key}/{name}")
            result[name] = float(metric)
    return result


def cohort_event(event: Event, cohort: Cohort) -> Event | None:
    if cohort == "public":
        return event
    if not all(isinstance(event.get(f"guard_{key}"), dict) for key in ("raw", "fitted", "jev")):
        return None
    return {**event, **{key: event[f"guard_{key}"] for key in ("raw", "fitted", "jev")}}


def prepare(events: Sequence[Event], run: str | None, panel: Panel = "full", cohort: Cohort = "public") -> Series:
    candidates = [
        str(event["run"])
        for event in events
        if isinstance(event.get("run"), str)
        and event["kind"] in ("training_started", "evaluation", "quick_evaluation", "run_restarted", "final_comparison")
    ]
    selected = run if run is not None else (candidates[-1] if candidates else "not started")
    if run is not None and run not in candidates:
        raise ValueError(f"No recorded training run named {run}")
    relevant = [event for event in events if event.get("run") == selected]
    steps = [event for event in relevant if event["kind"] == "training_step"]
    kind = "quick_evaluation" if panel == "quick" else "evaluation"
    evaluations: list[Event] = []
    for event in relevant:
        if event["kind"] != kind:
            continue
        if panel == "quick" and (
            event.get("diagnostic") is not True
            or event.get("selection_candidate") is not False
            or tuple(event.get(key) for key in ("development_rows", "guard_rows", "temperature_rows"))
            != (1000, 200, 700)
        ):
            raise ValueError("Quick panels require the declared 1000/200/700 diagnostic-only measurements")
        selected_event = cohort_event(event, cohort)
        if selected_event is not None:
            evaluations.append(selected_event)
    reference: dict[str, float] = {}
    for event in evaluations:
        current = metric_values(event, "jev")
        if reference and current != reference:
            raise ValueError("A fixed development comparison cannot change its Jev baseline mid-run")
        reference = current
    total = max(
        [int(number(event.get("total_steps"))) for event in relevant]
        + [int(number(event.get("step"))) for event in steps]
        + [1]
    )
    return {
        "run": selected,
        "panel": panel,
        "cohort": cohort,
        "events": list(events),
        "steps": steps,
        "evaluations": evaluations,
        "reference": reference,
        "total_steps": total,
    }


def research_history(events: Sequence[Event]) -> list[HistoryRow]:
    runs: dict[str, HistoryRow] = {}
    for event in events:
        run, kind = event.get("run"), str(event["kind"])
        if not isinstance(run, str) or kind not in RUN_EVENTS | {"pilot_recovery_verified"}:
            continue
        row = runs.setdefault(run, {"run": run, "step": 0, "status": "recorded", "detail": "", "recovery": ""})
        row["step"] = max(row["step"], int(number(event.get("step", event.get("steps")))))
        if kind == "run_restarted":
            previous = event.get("previous_run", "previous attempt")
            row["status"] = "restart planned"
            row["detail"] = f"After {previous}: {event.get('reason', 'separate recorded run')}"
        elif kind == "training_started":
            row["status"] = "running"
        elif kind in ("training_paused", "training_finished"):
            row["status"] = kind.removeprefix("training_")
        elif kind == "training_failed":
            phase = str(event.get("phase", ""))
            row["status"] = "restart failed" if phase.startswith("resume") else "training failed"
            row["detail"] = str(event.get("error", event.get("reason", "Failure recorded")))
            if event.get("updates_after_resume") == 0:
                row["detail"] += "; no additional optimizer update"
        elif kind == "pilot_recovery_verified" and event.get("logits_bitwise_equal") is True:
            row["recovery"] = f"Recovery matched {int(number(event.get('rows'))):,} development rows exactly."
    return list(runs.values())


def segments(events: Sequence[Event]) -> list[list[Event]]:
    result: list[list[Event]] = []
    previous = -1.0
    for event in events:
        step = number(event.get("step"))
        if not result or step <= previous:
            result.append([])
        result[-1].append(event)
        previous = step
    return result


def style_axis(axis: Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#d2d9d5")
    axis.tick_params(colors="#61726a", labelsize=9)
    axis.grid(axis="y", color="#edf1ee", linewidth=0.8)
    axis.set_axisbelow(True)


def trailing_mean(values: Sequence[float], window: int = 10) -> list[float]:
    return [sum(values[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for i in range(len(values))]


def milestone(event: Event) -> str:
    kind = str(event["kind"])
    task = event.get("task")
    if kind in ("task_started", "task_finished", "agent_task"):
        status = str(event.get("status", "started" if kind == "task_started" else "completed"))
        return f"{event.get('agent', 'Team')}: {str(task or kind).replace('_', ' ')} ({status})"
    if kind == "data_task_partitioned":
        return f"Partitioned {task} by source family"
    if kind.startswith("data_"):
        return f"{kind.removeprefix('data_').replace('_', ' ')}: {task}" if task else kind.replace("_", " ")
    if kind in {"evaluation", "quick_evaluation"}:
        values = metric_values(event, "fitted")
        prefix = "Quick diagnostic · " if kind == "quick_evaluation" else ""
        return f"{prefix}Step {int(number(event.get('step')))}: accuracy {values.get('accuracy', 0):.2%}; ECE {values.get('ece', 0):.4f}"
    if kind == "training_step":
        return f"Optimizer step {int(number(event.get('step')))}"
    if kind == "training_started":
        return f"Started {event.get('run', 'training')} from step {int(number(event.get('step')))}"
    if kind == "training_failed":
        phase = str(event.get("phase", "")).replace("_", " ")
        return f"{event.get('run', 'Training')}: {phase or 'training'} failed at step {int(number(event.get('step')))}"
    if kind == "run_restarted":
        return f"{event.get('previous_run', 'Previous run')} → {event.get('run', 'new run')}: {event.get('reason', 'restart recorded')}"
    if kind == "pilot_recovery_verified":
        return f"{event.get('run', 'Pilot')}: recovery checked on {int(number(event.get('rows'))):,} development rows"
    if kind in ("training_finished", "training_paused"):
        return f"{event.get('run', 'Training')}: {kind.removeprefix('training_')} at step {int(number(event.get('steps')))}"
    if kind == "public_evaluation":
        accuracy = metric_values(event, "metrics").get("accuracy")
        return (
            f"Known suite · update {int(number(event.get('step')))} · accuracy {accuracy:.2%}"
            if accuracy is not None
            else "Known-suite evaluation"
        )
    if kind == "final_comparison":
        return f"Locked test: {int(number(event.get('count'))):,} paired examples"
    if kind == "repeat_cancelled" and event.get("started") is False:
        return f"Cancelled {event.get('run', 'repeat')} before training; one full seed"
    return str(task or kind).replace("_", " ")


def active_series(events: Sequence[Event], cutoff: float, panel: Panel = "full", cohort: Cohort = "public") -> Series:
    visible = [event for event in events if timestamp(event) <= cutoff]
    transitions = [
        event
        for event in visible
        if event["kind"] in ("training_started", "run_restarted", "final_comparison")
        and isinstance(event.get("run"), str)
    ]
    return (
        prepare(events, str(transitions[-1]["run"]), panel, cohort)
        if transitions
        else prepare(visible, None, panel, cohort)
    )


def draw(series: Series, cutoff: float, *, width: int = 1920, height: int = 1080, all_runs: bool = False) -> Figure:
    figure = plt.figure(figsize=(width / 120, height / 120), dpi=120, facecolor="white")
    grid = figure.add_gridspec(
        3, 3, width_ratios=[1.05, 1.35, 1.2], left=0.045, right=0.97, top=0.83, bottom=0.11, hspace=0.70, wspace=0.35
    )
    activity = figure.add_subplot(grid[:, 0])
    loss_axis = figure.add_subplot(grid[:, 1])
    metric_axes = [figure.add_subplot(grid[index, 2]) for index in range(3)]
    visible = [event for event in series["events"] if timestamp(event) <= cutoff]
    steps = [event for event in series["steps"] if timestamp(event) <= cutoff]
    evaluations = [event for event in series["evaluations"] if timestamp(event) <= cutoff]
    final_events = (
        [event for event in visible if event["kind"] == "final_comparison" and event.get("run") == series["run"]]
        if series["panel"] == "full" and series["cohort"] == "public"
        else []
    )
    final = final_events[-1] if final_events else None
    final_effects = cast(dict[str, JSONValue], final.get("effects", {})) if final else {}
    first = timestamp(series["events"][0])
    elapsed = max(0, int(cutoff - first))
    heading = "AUTOJEV / LOCKED TEST" if final else "AUTOJEV / SUPERVISED LEARNING"
    if all_runs and not final:
        heading = "AUTOJEV / RESEARCH" if series["run"] == "not started" else f"AUTOJEV / {series['run'].upper()}"
    if series["panel"] == "quick":
        heading = f"AUTOJEV / QUICK {series['cohort'].upper()} DIAGNOSTIC"
    elif series["cohort"] == "guard":
        heading = "AUTOJEV / FULL SYNTHETIC GUARD"
    figure.text(0.045, 0.93, heading, color=GREEN, fontsize=22, weight="bold")
    identity = (
        f"{series['run']} · selected update {final.get('selected_step', 'unrecorded')} · T={number(final.get('temperature')):.4f}"
        if final
        else series["run"]
    )
    figure.text(
        0.045,
        0.885,
        f"{identity}  ·  {elapsed // 3600:02d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d} actual elapsed",
        color="#61726a",
        fontsize=11,
    )
    if visible:
        figure.text(
            0.97,
            0.925,
            str(visible[-1]["timestamp"])[:19].replace("T", " ") + " UTC",
            ha="right",
            fontsize=10,
            color=GRAY,
        )

    activity.set_axis_off()
    activity.set_title("Recorded work", loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    milestones = [event for event in visible if event["kind"] not in ("training_step", "resume_saved")]
    history = research_history(visible)
    show_history = any("failed" in row["status"] or row["detail"] for row in history)
    recent = milestones[-3:] if show_history else milestones[-7:]
    cursor = 0.95
    for event in recent:
        label = textwrap.fill(milestone(event), width=36, max_lines=3, placeholder="…")
        color = "#a45835" if event["kind"] == "training_failed" else GREEN
        activity.text(0, cursor, "●", color=color, fontsize=10, va="top", transform=activity.transAxes)
        activity.text(
            0.07, cursor, label, color=INK, fontsize=10, va="top", linespacing=1.4, transform=activity.transAxes
        )
        cursor -= 0.052 + 0.035 * len(label.splitlines())
    if not recent:
        activity.text(0, 0.9, "Waiting for recorded work", color=GRAY, transform=activity.transAxes)
    if show_history:
        activity.text(
            0, 0.41, "Research history", fontweight="bold", color=INK, fontsize=10, transform=activity.transAxes
        )
        cursor = 0.36
        for row in history[-3:]:
            color = "#a45835" if "failed" in row["status"] else GREEN
            activity.text(
                0,
                cursor,
                f"{row['run']} · update {row['step']} · {row['status']}",
                fontsize=9,
                color=color,
                va="top",
                transform=activity.transAxes,
            )
            detail = row["recovery"] or row["detail"]
            if detail:
                activity.text(
                    0,
                    cursor - 0.035,
                    textwrap.fill(detail, width=47, max_lines=2, placeholder="…"),
                    fontsize=8,
                    color=GRAY,
                    va="top",
                    transform=activity.transAxes,
                )
            cursor -= 0.11
    repeat_cancelled = any(event["kind"] == "repeat_cancelled" for event in visible)
    if repeat_cancelled:
        activity.text(
            0,
            0.095,
            "One full seed; planned repeat cancelled.\nSeed-to-seed robustness was not measured.",
            color=GRAY,
            fontsize=8.5,
            linespacing=1.5,
            transform=activity.transAxes,
        )
    activity.text(
        0,
        -0.05,
        "Research attempts remain in the record.\nCurves show the selected run only.",
        color=GRAY,
        fontsize=8.5,
        linespacing=1.5,
        transform=activity.transAxes,
    )

    style_axis(loss_axis, "Training cross-entropy ↓")
    loss_axis.set_xlim(0, series["total_steps"])
    maximum = max([number(event.get("loss")) for event in series["steps"]] + [1.0])
    loss_axis.set_ylim(0, maximum * 1.10)
    loss_axis.set_xlabel("Optimizer update", color="#61726a", fontsize=10)
    for index, segment in enumerate(segments(steps)):
        x = [number(event.get("step")) for event in segment]
        y = [number(event.get("loss")) for event in segment]
        marker = "o" if len(segment) == 1 else None
        loss_axis.plot(
            x,
            y,
            color=PALE,
            linewidth=1.5,
            marker=marker,
            markersize=3.5,
            label="Recorded loss" if index == 0 else None,
        )
        loss_axis.plot(
            x,
            trailing_mean(y),
            color=GREEN,
            linewidth=2.1,
            marker=marker,
            markersize=3.5,
            label="Trailing 10-update mean" if index == 0 else None,
        )
    if steps:
        loss_axis.legend(frameon=False, fontsize=8, loc="upper right")
        loss_axis.text(
            0.04,
            0.05,
            f"Latest loss  {number(steps[-1].get('loss')):.4f}",
            color=GREEN,
            fontsize=12,
            weight="bold",
            transform=loss_axis.transAxes,
        )
    else:
        loss_axis.text(
            0.5,
            0.5,
            "Training measurements\nwill appear when recorded",
            ha="center",
            va="center",
            color=GRAY,
            fontsize=11,
            transform=loss_axis.transAxes,
        )
    if final and isinstance(final.get("selected_step"), int):
        loss_axis.axvline(number(final["selected_step"]), color=GRAY, linestyle=":", linewidth=1)
        loss_axis.text(
            0.04,
            0.095,
            f"Frozen checkpoint: update {final['selected_step']}",
            color=GRAY,
            fontsize=9,
            transform=loss_axis.transAxes,
        )

    for axis, name, title in zip(
        metric_axes,
        ("accuracy", "ece", "brier"),
        ("Overall accuracy ↑", "Calibration error ↓", "Brier probability score ↓"),
        strict=True,
    ):
        style_axis(axis, title)
        if final:
            effect = cast(dict[str, JSONValue], final_effects[name])
            local, reference_value = number(effect["local"]), number(effect["reference"])
            axis.bar([0, 1], [local, reference_value], width=0.55, color=[GREEN, GRAY])
            axis.set_xticks([0, 1], ["Local", "Jev"])
            axis.set_xlim(-0.7, 1.7)
            if name == "accuracy":
                axis.set_ylim(0, 1)
                axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            else:
                axis.set_ylim(0, max(local, reference_value + MARGIN, MARGIN) * 1.2)
                axis.axhline(reference_value + MARGIN, color="#a0b09e", linestyle=":", linewidth=1)
            for position, value in enumerate((local, reference_value)):
                axis.annotate(
                    f"{value:.2%}" if name == "accuracy" else f"{value:.4f}",
                    (position, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=INK,
                )
            interval = cast(list[JSONValue], effect["difference_ci95"])
            scale, unit = (100, " pp") if name == "accuracy" else (1, "")
            difference = number(effect["difference"]) * scale
            low, high = number(interval[0]) * scale, number(interval[1]) * scale
            axis.text(
                0.5,
                -0.34,
                f"Δ {difference:+.3f}{unit} · paired 95% [{low:+.3f}, {high:+.3f}]",
                ha="center",
                fontsize=8,
                color=GRAY,
                transform=axis.transAxes,
            )
            continue
        axis.set_xlim(0, series["total_steps"])
        probability_modes = ("fitted", "raw") if name != "accuracy" else ("fitted",)
        all_values = [
            metric_values(event, mode)[name]
            for event in series["evaluations"]
            for mode in probability_modes
            if name in metric_values(event, mode)
        ]
        reference = series["reference"].get(name)
        if reference is not None:
            all_values.append(reference)
        if name == "accuracy":
            axis.set_ylim(max(0, min(all_values, default=0.5) - 0.045), min(1, max(all_values, default=0.95) + 0.035))
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            axis.set_ylim(0, max(all_values + [MARGIN], default=0.1) * 1.2 + MARGIN)
        if reference is not None and evaluations:
            axis.axhline(reference, color=GRAY, linewidth=1.2, linestyle="--", label="Jev")
            if name != "accuracy":
                axis.axhspan(0, reference + MARGIN, color=PALE, alpha=0.25)
                axis.axhline(
                    reference + MARGIN, color="#a0b09e", linewidth=0.9, linestyle=":", label="Jev + 0.010 margin"
                )
        for index, segment in enumerate(segments(evaluations)):
            x = [number(event.get("step")) for event in segment]
            y = [metric_values(event, "fitted")[name] for event in segment]
            if name != "accuracy":
                raw = [metric_values(event, "raw")[name] for event in segment]
                axis.plot(
                    x,
                    raw,
                    color=RAW,
                    linewidth=1.4,
                    linestyle=":",
                    marker="x",
                    markersize=4,
                    label="Raw (diagnostic)" if index == 0 else None,
                )
            label = "Local" if name == "accuracy" else "Fitted"
            axis.plot(x, y, color=GREEN, linewidth=1.8, marker="o", markersize=3.5, label=label if index == 0 else None)
        if evaluations:
            axis.legend(
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.96,
                fancybox=False,
                fontsize=7,
                loc="upper right",
                ncol=1 if name == "accuracy" else 2,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No evaluation recorded",
                ha="center",
                va="center",
                color=GRAY,
                fontsize=9,
                transform=axis.transAxes,
            )
    if final:
        status = "met" if final.get("empirical_target_met") is True else "not met"
        supported = "established" if final.get("statistically_supported") is True else "not established"
        footer = f"Locked test: {int(number(final.get('count'))):,} examples · observed accuracy/calibration target {status} · paired statistical support {supported}"
        if repeat_cancelled:
            footer += " · one training seed"
    else:
        metric_axes[-1].set_xlabel(
            f"Optimizer update · {'quick ' if series['panel'] == 'quick' else ''}{'synthetic guard' if series['cohort'] == 'guard' else 'fixed development fold'}",
            color="#61726a",
            fontsize=9,
        )
        footer = (
            QUICK_NOTE + f" · plotted: {series['cohort']} · Jev uses the same subset"
            if series["panel"] == "quick"
            else "Measured checkpoints only · raw probabilities: diagnostic · selection/final comparison: fitted (separate temperature fold) · hard reference labels"
        )
    figure.text(0.045, 0.035, footer, color=GRAY, fontsize=9)
    return figure


def dashboard(series: Series, destination: Path, *, watch: bool = False) -> None:
    keep = {
        "timestamp",
        "kind",
        "run",
        "task",
        "agent",
        "status",
        "step",
        "steps",
        "total_steps",
        "loss",
        "fitted",
        "raw",
        "jev",
        "eligible",
        "selected_step",
        "temperature",
        "count",
        "effects",
        "empirical_target_met",
        "statistically_supported",
        "previous_run",
        "reason",
        "error",
        "phase",
        "from_base",
        "recipe_changed",
        "updates_after_resume",
        "rows",
        "logits_bitwise_equal",
        "calibration_margin",
        "freeze_sha256",
        "metrics",
        "started",
        "guard_raw",
        "guard_fitted",
        "guard_jev",
        "diagnostic",
        "selection_candidate",
        "development_rows",
        "guard_rows",
        "temperature_rows",
    }
    events = [{key: value for key, value in event.items() if key in keep} for event in series["events"]]
    for event in events:
        if "metrics" in event:
            event["metrics"] = cast(JSONValue, metric_values(event, "metrics"))
    payload = json.dumps(
        {
            "events": events,
            "run": series["run"],
            "panel": series["panel"],
            "cohort": series["cohort"],
            "total_steps": series["total_steps"],
            "watch": watch,
        },
        ensure_ascii=False,
        allow_nan=False,
    ).replace("<", "\\u003c")
    content = DASHBOARD.replace("__PAYLOAD__", payload).replace("__TITLE__", html.escape(series["run"]))
    destination.write_text(content)
    if watch:
        temporary = destination.with_name("live.json.tmp")
        temporary.write_text(payload)
        temporary.replace(destination.with_name("live.json"))


DASHBOARD = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoJev · __TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f8f6;color:#273431;font:15px system-ui,sans-serif}main{max-width:1500px;margin:35px auto;padding:28px;background:white;border:1px solid #e2e8e2;border-radius:18px}h1{color:#286b4d;font-size:26px;margin:0 0 9px}p,.muted{color:#718078}.controls{display:flex;gap:14px;align-items:center;margin:24px 0}button,select{background:white;border:1px solid #b9c9bd;border-radius:7px;padding:9px;color:#273431}input{flex:1;accent-color:#286b4d}.panels{display:grid;grid-template-columns:1fr 1.5fr 1.3fr;gap:28px}h2{font-size:16px}canvas{width:100%;height:185px;display:block}#loss{height:535px}#tasks{padding:0;list-style:none}#tasks li{border-left:2px solid #bed4c4;padding:0 0 18px 13px;line-height:1.5}.metric{border-bottom:1px solid #eef2ef;padding-bottom:14px}.stat{color:#286b4d;font-weight:650}footer{font-size:12px;margin-top:24px;color:#718078;line-height:1.6}.legend{font-size:12px;color:#718078}#table{width:100%;border-collapse:collapse;font-size:13px;margin-top:18px}td,th{text-align:right;border-bottom:1px solid #eef2ef;padding:8px}td:first-child,th:first-child{text-align:left}details{margin-top:25px}@media(max-width:850px){main{margin:0;border:0;padding:22px}.panels{grid-template-columns:1fr}#loss{height:280px}.controls{flex-wrap:wrap}input{min-width:180px}}
</style><main><h1>AUTOJEV / SUPERVISED LEARNING</h1><div class="muted" id="status"></div>
<style>#history{margin-top:28px;border-top:1px solid #e2e8e2;padding-top:14px}#history h3{font-size:14px}#history-list p{font-size:12px;margin:5px 0 15px;line-height:1.5}.failed{color:#a45835}</style>
<div class="controls"><button id="play">Play</button><input id="time" type="range" min="0" step="1"><label>Run <select id="run"></select></label><label>Panel <select id="panel"><option value="full">Full evaluations</option><option value="quick">Quick diagnostic</option></select></label><label>Cohort <select id="cohort"><option value="public">Public</option><option value="guard">Synthetic guard</option></select></label><label>Probabilities <select id="mode"><option value="fitted">Temperature fitted</option><option value="raw">Raw</option></select></label></div>
<p class="legend" id="panel-note"></p>
<div class="panels"><section><h2>Recorded work</h2><ul id="tasks"></ul><section id="history" hidden><h3>Research history</h3><div id="history-list"></div><p class="legend">Each run has its own curve. The setup pilot is not part of the full run.</p></section></section><section><h2>Training cross-entropy ↓</h2><canvas id="loss"></canvas><div class="legend">Light: recorded loss · dark: trailing 10-update mean</div></section><section>
<div class="metric"><h2>Overall accuracy ↑ <span class="stat" id="accuracy-value"></span></h2><canvas id="accuracy"></canvas></div>
<div class="metric"><h2>Calibration error ↓ <span class="stat" id="ece-value"></span></h2><canvas id="ece"></canvas></div>
<div class="metric"><h2>Brier probability score ↓ <span class="stat" id="brier-value"></span></h2><canvas id="brier"></canvas></div>
<div class="legend">Green: local · dashed: Jev on the same selected cohort · shaded: Jev + 0.010 margin</div></section></div>
<details><summary>Measured checkpoints</summary><table id="table"><thead><tr><th>Step</th><th>Accuracy</th><th>ECE</th><th>Brier</th><th>Temperature</th><th>Eligible</th></tr></thead><tbody></tbody></table></details>
<section id="final" hidden><h2>Locked test comparison</h2><p id="final-status"></p><table id="final-table"><thead><tr><th>Metric</th><th>Local</th><th>Jev</th><th>Difference</th><th>Paired 95% interval</th></tr></thead><tbody></tbody></table></section>
<footer id="report-note">Every point comes from a recorded evaluation on the fixed development fold. Temperature is fitted on a separate fold and does not change accuracy. Raw curves remain available. Repeated steps after a restart create an explicit curve break. Research attempts are separate runs; no best-so-far substitution or synthetic evaluation points are used.</footer></main>
<script>
let data=__PAYLOAD__,events=data.events;const slider=document.querySelector('#time'),selector=document.querySelector('#run'),mode=document.querySelector('#mode'),panel=document.querySelector('#panel'),cohort=document.querySelector('#cohort');panel.value=data.panel||'full';cohort.value=data.cohort||'public';const fullNote=document.querySelector('#report-note').textContent;
function refreshRuns(selected){const runs=[...new Set(events.filter(e=>e.run&&['training_started','evaluation','quick_evaluation','run_restarted','final_comparison'].includes(e.kind)).map(e=>e.run))];if(!runs.length)runs.push('not started');selector.replaceChildren();for(const run of runs){const option=document.createElement('option');option.value=run;option.textContent=run;selector.append(option)}selector.value=runs.includes(selected)?selected:data.run}refreshRuns(data.run);slider.max=events.length-1;slider.value=events.length-1;
function label(e){
if(e.kind==='training_failed')return `${e.run}: ${(e.phase||'training').replaceAll('_',' ')} failed at step ${e.step}`;
if(e.kind==='run_restarted')return `${e.previous_run||'Previous run'} → ${e.run}: ${e.reason||'restart recorded'}`;
if(e.kind==='pilot_recovery_verified')return `${e.run}: recovery checked on ${e.rows.toLocaleString()} development rows`;
if(e.kind==='public_evaluation')return Number.isFinite(e.metrics?.accuracy)?`Known suite · update ${e.step} · accuracy ${(e.metrics.accuracy*100).toFixed(2)}%`:'Known-suite evaluation';
if(['evaluation','quick_evaluation'].includes(e.kind))return `${e.kind==='quick_evaluation'?'Quick diagnostic · ':''}Step ${e.step}: accuracy ${(e.fitted.accuracy*100).toFixed(2)}%; ECE ${e.fitted.ece.toFixed(4)}`;
if(e.kind==='data_task_partitioned')return `Partitioned ${e.task} by source family`;
if(e.kind.startsWith('data_')&&e.task)return `${e.kind.slice(5).replaceAll('_',' ')}: ${e.task}`;
return `${e.agent?e.agent+': ':''}${String(e.task||e.kind).replaceAll('_',' ')}${e.status?' ('+e.status+')':''}`}
function historyRows(visible){const result=new Map();for(const e of visible){if(!e.run||!['training_started','training_step','evaluation','quick_evaluation','training_paused','training_finished','training_failed','run_restarted','pilot_recovery_verified'].includes(e.kind))continue;if(!result.has(e.run))result.set(e.run,{run:e.run,step:0,status:'recorded',detail:'',recovery:''});const row=result.get(e.run);row.step=Math.max(row.step,e.step||e.steps||0);if(e.kind==='run_restarted'){row.status='restart planned';row.detail=`After ${e.previous_run||'previous attempt'}: ${e.reason||'separate recorded run'}`}else if(e.kind==='training_started')row.status='running';else if(['training_paused','training_finished'].includes(e.kind))row.status=e.kind.slice(9);else if(e.kind==='training_failed'){row.status=(e.phase||'').startsWith('resume')?'restart failed':'training failed';row.detail=e.error||e.reason||'Failure recorded';if(e.updates_after_resume===0)row.detail+='; no additional optimizer update'}else if(e.kind==='pilot_recovery_verified'&&e.logits_bitwise_equal)row.recovery=`Recovery matched ${e.rows.toLocaleString()} development rows exactly.`}return [...result.values()]}
function split(points){const groups=[];for(const p of points){if(!groups.length||p[0]<=groups[groups.length-1].at(-1)[0])groups.push([]);groups.at(-1).push(p)}return groups}
function chart(id,groups,range,reference,margin,xmax,percent=false){const c=document.getElementById(id),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const l=46,r=w-14,t=12,b=h-31;const x=v=>l+v/xmax*(r-l),y=v=>b-(v-range[0])/(range[1]-range[0])*(b-t);ctx.font='11px system-ui';ctx.fillStyle='#718078';ctx.strokeStyle='#edf1ee';for(let i=0;i<=3;i++){const v=range[0]+i/3*(range[1]-range[0]),yy=y(v);ctx.beginPath();ctx.moveTo(l,yy);ctx.lineTo(r,yy);ctx.stroke();ctx.fillText(percent?(100*v).toFixed(0)+'%':v.toFixed(2),2,yy+4)}if(reference!==null){if(margin){ctx.fillStyle='#c7dccb55';ctx.fillRect(l,y(reference+margin),r-l,b-y(reference+margin));ctx.strokeStyle='#a0b09e';ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(l,y(reference+margin));ctx.lineTo(r,y(reference+margin));ctx.stroke()}ctx.strokeStyle='#87938f';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(l,y(reference));ctx.lineTo(r,y(reference));ctx.stroke();ctx.setLineDash([])}for(const group of groups){ctx.strokeStyle=group.color;ctx.fillStyle=group.color;ctx.lineWidth=group.width||2;for(const part of split(group.points)){ctx.beginPath();part.forEach((p,i)=>i?ctx.lineTo(x(p[0]),y(p[1])):ctx.moveTo(x(p[0]),y(p[1])));ctx.stroke();if(group.dots||part.length===1)for(const p of part){ctx.beginPath();ctx.arc(x(p[0]),y(p[1]),3,0,Math.PI*2);ctx.fill()}}}ctx.fillStyle='#718078';ctx.fillText('0',l,b+21);ctx.fillText(String(xmax),r-20,b+21);if(groups.every(g=>!g.points.length)){ctx.fillStyle='#87938f';ctx.fillText('No measurements recorded',l+10,(t+b)/2)}}
function panelEvents(items){const kind=panel.value==='quick'?'quick_evaluation':'evaluation';return items.filter(e=>e.kind===kind).map(e=>{if(panel.value==='quick'&&(e.diagnostic!==true||e.selection_candidate!==false||e.development_rows!==1000||e.guard_rows!==200||e.temperature_rows!==700))throw Error('Quick panel requires the declared 1000/200/700 diagnostic-only measurements');return cohort.value==='public'?e:e.guard_raw&&e.guard_fitted&&e.guard_jev?{...e,raw:e.guard_raw,fitted:e.guard_fitted,jev:e.guard_jev}:null}).filter(Boolean)}
function render(){const position=+slider.value,visible=events.slice(0,position+1),run=selector.value,all=events.filter(e=>e.run===run),current=visible.filter(e=>e.run===run),steps=current.filter(e=>e.kind==='training_step'),evals=panelEvents(current),allEvals=panelEvents(all),xmax=Math.max(1,...all.map(e=>e.total_steps||e.step||0));const last=visible.at(-1);document.querySelector('#status').textContent=`${run} · ${last.timestamp.replace('T',' ').slice(0,19)} UTC · ${position+1}/${events.length} recorded events`;
const quick=panel.value==='quick';document.querySelector('h1').textContent=quick?`AUTOJEV / QUICK ${cohort.value.toUpperCase()} DIAGNOSTIC`:cohort.value==='guard'?'AUTOJEV / FULL SYNTHETIC GUARD':'AUTOJEV / SUPERVISED LEARNING';document.querySelector('#panel-note').textContent=quick?`Quick diagnostic: 1,000 public / 200 guard / 700 temperature-fit rows · plotted: ${cohort.value} · never selects checkpoints`:`Full evaluations · plotted: ${cohort.value}${cohort.value==='guard'?' · separate synthetic guard cohort':''}`;document.querySelector('#report-note').textContent=quick?'Every point is a recorded quick evaluation. Jev is evaluated on the same fixed subset. Temperature is fitted separately on the 700-row diagnostic fold. These measurements never select checkpoints and are never connected to full-evaluation curves.':fullNote;
const history=historyRows(visible),showHistory=history.some(row=>row.status.includes('failed')||row.detail);document.querySelector('#history').hidden=!showHistory;const historyList=document.querySelector('#history-list');historyList.replaceChildren();for(const row of history){const title=document.createElement('strong');title.textContent=`${row.run} · update ${row.step} · ${row.status}`;if(row.status.includes('failed'))title.className='failed';const detail=document.createElement('p');detail.textContent=[row.detail,row.recovery].filter(Boolean).join(' ');historyList.append(title,detail)}
const tasks=document.querySelector('#tasks');tasks.replaceChildren();for(const event of visible.filter(e=>!['training_step','resume_saved'].includes(e.kind)).slice(showHistory?-3:-7)){const li=document.createElement('li');li.textContent=label(event);if(event.kind==='training_failed')li.className='failed';tasks.append(li)}const losses=steps.map(e=>[e.step,e.loss]),smoothed=[];for(const part of split(losses))for(let i=0;i<part.length;i++){const tail=part.slice(Math.max(0,i-9),i+1);smoothed.push([part[i][0],tail.reduce((s,p)=>s+p[1],0)/tail.length])}chart('loss',[{points:losses,color:'#c7dccb',width:1.5},{points:smoothed,color:'#286b4d'}],[0,1.1*Math.max(1,...all.filter(e=>e.kind==='training_step').map(e=>e.loss))],null,0,xmax);
for(const metric of ['accuracy','ece','brier']){const reference=evals.length&&Number.isFinite(evals[0].jev?.[metric])?evals[0].jev[metric]:null,values=allEvals.map(e=>e[mode.value][metric]);if(reference!==null)values.push(reference);const range=metric==='accuracy'?[Math.max(0,(values.length?Math.min(...values):.5)-.045),Math.min(1,(values.length?Math.max(...values):.95)+.035)]:[0,1.2*Math.max(.01,...values)+.01];chart(metric,[{points:evals.map(e=>[e.step,e[mode.value][metric]]),color:'#286b4d',dots:true}],range,reference,metric==='accuracy'?0:.01,xmax,metric==='accuracy');document.getElementById(metric+'-value').textContent=evals.length?(metric==='accuracy'?(evals.at(-1)[mode.value][metric]*100).toFixed(2)+'%':evals.at(-1)[mode.value][metric].toFixed(4)):''}const body=document.querySelector('#table tbody');body.replaceChildren();for(const e of evals){const tr=document.createElement('tr');for(const value of [e.step,(e[mode.value].accuracy*100).toFixed(2)+'%',e[mode.value].ece.toFixed(4),e[mode.value].brier.toFixed(4),e.temperature.toFixed(4),panel.value==='quick'?'diagnostic only':e.eligible?'yes':'no']){const td=document.createElement('td');td.textContent=String(value);tr.append(td)}body.append(tr)}
const final=panel.value==='full'&&cohort.value==='public'?current.filter(e=>e.kind==='final_comparison').at(-1):null;document.querySelector('#final').hidden=!final;if(final){document.querySelector('#final-status').textContent=`Frozen ${final.run}, update ${final.selected_step??'unrecorded'}, temperature ${Number(final.temperature).toFixed(4)}. ${final.count.toLocaleString()} fixed examples. Observed target ${final.empirical_target_met?'met':'not met'}; paired statistical support ${final.statistically_supported?'established':'not established'}. This table always uses the deployed temperature-fitted probabilities; the raw/fitted control affects only development curves.`;const tbody=document.querySelector('#final-table tbody');tbody.replaceChildren();for(const metric of ['accuracy','ece','brier']){const e=final.effects[metric],s=metric==='accuracy'?100:1,unit=metric==='accuracy'?' pp':'',suffix=metric==='accuracy'?'%':'';const tr=document.createElement('tr');for(const value of [metric,(e.local*s).toFixed(4)+suffix,(e.reference*s).toFixed(4)+suffix,(e.difference*s).toFixed(4)+unit,e.difference_ci95.map(x=>(x*s).toFixed(4)).join(' to ')+unit]){const td=document.createElement('td');td.textContent=String(value);tr.append(td)}tbody.append(tr)}}}
let timer=null;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play'}else{if(+slider.value>=+slider.max)slider.value=0;document.querySelector('#play').textContent='Pause';timer=setInterval(()=>{slider.value=+slider.value+1;render();if(+slider.value>=+slider.max){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play'}},Math.max(60,60000/events.length))}};slider.oninput=render;selector.onchange=render;mode.onchange=render;panel.onchange=render;cohort.onchange=render;addEventListener('resize',render);render();
async function refreshLive(){try{const response=await fetch('live.json',{cache:'no-store'});if(!response.ok)return;const updated=await response.json();if(JSON.stringify(updated)===JSON.stringify(data))return;const selected=selector.value,position=+slider.value,follow=position===+slider.max&&!timer;data=updated;events=data.events;refreshRuns(selected);slider.max=events.length-1;slider.value=follow?slider.max:Math.min(position,+slider.max);render()}catch(error){document.querySelector('#status').title='Live refresh unavailable; showing the last recorded snapshot'}}
if(data.watch&&location.protocol.startsWith('http'))setInterval(refreshLive,10000);
</script></html>"""


def film(
    series: Series, output: Path, *, duration: int, fps: int, width: int, height: int, all_runs: bool = False
) -> None:
    executable = cast(Callable[[], str], importlib.import_module("imageio_ffmpeg").get_ffmpeg_exe)()
    times = [timestamp(event) for event in series["events"]]
    selected_times = [
        timestamp(event)
        for event in series["events"]
        if (event["kind"] == "training_started" if all_runs else event.get("run") == series["run"])
    ]
    beginning, end = times[0], times[-1]
    training_start = selected_times[0] if selected_times else beginning
    frames = duration * fps
    command = [
        executable,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with (
        output.with_suffix(".ffmpeg.log").open("w") as log,
        subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log) as writer,
    ):
        assert writer.stdin is not None
        try:
            for index in range(frames):
                fraction = index / max(1, frames - 1)
                if fraction < 0.15 and training_start > beginning:
                    cutoff = beginning + fraction / 0.15 * (training_start - beginning)
                else:
                    progress = (
                        min(1, max(0, (fraction - 0.15) / 0.80))
                        if training_start > beginning
                        else min(1, fraction / 0.95)
                    )
                    cutoff = training_start + progress * (end - training_start)
                current = (
                    active_series(series["events"], cutoff, series["panel"], series["cohort"]) if all_runs else series
                )
                figure = draw(current, cutoff, width=width, height=height, all_runs=all_runs)
                canvas = FigureCanvasAgg(figure)
                cast(Callable[[], None], canvas.draw)()
                frame = np.asarray(cast(Callable[[], memoryview], canvas.buffer_rgba)())[:, :, :3].copy()
                writer.stdin.write(frame.tobytes())
                plt.close(figure)
                if index % (fps * 5) == 0:
                    print(f"Rendered {index}/{frames} frames", flush=True)
        finally:
            writer.stdin.close()
        status = writer.wait()
        if status:
            raise RuntimeError(f"Video encoder exited {status}; see {output.with_suffix('.ffmpeg.log')}")


class Arguments(argparse.Namespace):
    events: Path
    output: Path
    run: str | None
    panel: Panel
    cohort: Cohort
    video: bool
    all_runs: bool
    watch: bool
    duration: int
    fps: int
    width: int
    height: int


def write_report(args: Arguments, source_bytes: bytes) -> None:
    events = parse_events(source_bytes.decode())
    series = (
        active_series(events, timestamp(events[-1]), args.panel, args.cohort)
        if args.all_runs
        else prepare(events, args.run, args.panel, args.cohort)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    dashboard(series, args.output / "index.html", watch=args.watch)
    figure = draw(series, timestamp(events[-1]), width=args.width, height=args.height, all_runs=args.all_runs)
    figure.savefig(args.output / "training.png", dpi=120)
    svg = args.output / "training.svg"
    figure.savefig(svg)
    svg.write_text("\n".join(line.rstrip(" \t") for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)
    provenance = {
        "events": str(args.events),
        "events_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "events_bytes": len(source_bytes),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run": series["run"],
        "panel": series["panel"],
        "cohort": series["cohort"],
        "event_count": len(events),
        "last_event": events[-1]["timestamp"],
        "video": args.video,
        "duration_seconds": args.duration if args.video else None,
        "frames_per_second": args.fps if args.video else None,
        "all_runs": args.all_runs,
        "replay": "15% research, 80% chronological run history, 5% final hold; each run has separate curves"
        if args.all_runs
        else "15% research, 80% selected-run time, 5% final hold; only recorded events are revealed",
    }
    if args.video:
        film(
            series,
            args.output / "autojev.mp4",
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
            all_runs=args.all_runs,
        )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Report: {args.output / 'index.html'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded work, training loss and evaluation measurements as a report or film."
    )
    parser.add_argument("--events", type=Path, default=Path("progress/events.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("progress/report"))
    parser.add_argument("--run", help="Show one recorded training run; defaults to the latest")
    parser.add_argument(
        "--panel",
        choices=("full", "quick"),
        default="full",
        help="Keep full selection evaluations and quick diagnostics in separate plots",
    )
    parser.add_argument(
        "--cohort",
        choices=("public", "guard"),
        default="public",
        help="Use this cohort's own measurements and matching Jev reference",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Video follows recorded run starts and returns to the frozen selected run at final comparison",
    )
    parser.add_argument(
        "--watch", action="store_true", help="Refresh the report every 10 seconds when recorded events change"
    )
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(namespace=Arguments())
    if min(args.duration, args.fps) < 1 or args.width < 960 or args.height < 540 or args.width % 2 or args.height % 2:
        parser.error("Use positive duration/fps and even dimensions of at least 960×540")
    if args.watch and args.video:
        parser.error("Use --watch for the live dashboard or --video for a fixed replay")
    if args.all_runs and (not args.video or args.run is not None):
        parser.error("Use --all-runs with --video and without --run")
    previous: bytes | None = None
    while True:
        source_bytes = args.events.read_bytes()
        if source_bytes != previous:
            write_report(args, source_bytes)
            previous = source_bytes
        if not args.watch:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
