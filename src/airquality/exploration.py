"""Exploratory data analysis using Plotly.

Every figure is derived from the supplied dataframe. Functions return
plotly.graph_objects.Figure (or None with a message when data are insufficient).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .config import SEASON_ORDER


@dataclass
class FigureResult:
    figure: Optional[go.Figure]
    message: str


def _empty_figure(message: str) -> FigureResult:
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "#666"},
            }
        ],
        margin={"t": 40, "b": 40, "l": 40, "r": 40},
        template="plotly_white",
        height=400,
    )
    return FigureResult(None if False else fig, message)


def _require(df: pd.DataFrame, columns: list[str]) -> tuple[bool, str]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return False, f"Missing column(s): {', '.join(missing)}"
    return True, ""


def distribution_histogram(df: pd.DataFrame, column: str, title: Optional[str] = None) -> FigureResult:
    ok, msg = _require(df, [column])
    if not ok:
        return _empty_figure(msg)
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return _empty_figure(f"No numeric data for '{column}'.")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=s, name=column, marker_color="#1f77b4", opacity=0.75)
    )
    fig.update_layout(
        title=title or f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.1,
        height=450,
    )
    return FigureResult(fig, f"{len(s)} non-missing values.")


def aqi_distribution(df: pd.DataFrame, aqi_column: str = "aqi") -> FigureResult:
    return distribution_histogram(df, aqi_column, title="AQI Distribution")


def time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    location_column: Optional[str] = None,
    title: Optional[str] = None,
) -> FigureResult:
    ok, msg = _require(df, [date_column, value_column])
    if not ok:
        return _empty_figure(msg)
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column, value_column])
    if work.empty:
        return _empty_figure("No valid date/value pairs for time series.")
    work = work.sort_values(date_column)

    fig = go.Figure()
    if location_column and location_column in work.columns:
        for loc in sorted(work[location_column].dropna().unique().tolist())[:12]:
            sub = work[work[location_column] == loc]
            fig.add_trace(
                go.Scatter(
                    x=sub[date_column],
                    y=sub[value_column],
                    mode="lines+markers",
                    name=str(loc),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=work[date_column],
                y=work[value_column],
                mode="lines+markers",
                name=value_column,
            )
        )
    fig.update_layout(
        title=title or f"{value_column} over time",
        xaxis_title="Date",
        yaxis_title=value_column,
        template="plotly_white",
        height=450,
        legend_title_text="Location" if location_column else None,
    )
    return FigureResult(fig, f"{len(work)} points plotted.")


def box_by_location(
    df: pd.DataFrame,
    value_column: str,
    location_column: str,
    title: Optional[str] = None,
) -> FigureResult:
    ok, msg = _require(df, [value_column, location_column])
    if not ok:
        return _empty_figure(msg)
    work = df.dropna(subset=[value_column, location_column])
    if work.empty:
        return _empty_figure("No data for location comparison.")
    fig = go.Figure()
    for loc in sorted(work[location_column].unique().tolist()):
        sub = work[work[location_column] == loc][value_column]
        fig.add_trace(go.Box(y=sub, name=str(loc), boxmean="sd"))
    fig.update_layout(
        title=title or f"{value_column} by location",
        xaxis_title="Location",
        yaxis_title=value_column,
        template="plotly_white",
        height=450,
    )
    return FigureResult(fig, f"{work[location_column].nunique()} location(s).")


def monthly_profile(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: Optional[str] = None,
) -> FigureResult:
    ok, msg = _require(df, [date_column, value_column])
    if not ok:
        return _empty_figure(msg)
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column, value_column])
    if work.empty:
        return _empty_figure("No data for monthly profile.")
    work["month"] = work[date_column].dt.month
    monthly = work.groupby("month")[value_column].mean().reindex(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=[month_names[m - 1] for m in monthly.index], y=monthly.values, marker_color="#2ca02c")
    )
    fig.update_layout(
        title=title or f"Monthly mean {value_column}",
        xaxis_title="Month",
        yaxis_title=f"Mean {value_column}",
        template="plotly_white",
        height=450,
    )
    return FigureResult(fig, f"{len(work)} observations.")


def seasonal_profile(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: Optional[str] = None,
) -> FigureResult:
    ok, msg = _require(df, [date_column, value_column])
    if not ok:
        return _empty_figure(msg)
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column, value_column])
    if work.empty:
        return _empty_figure("No data for seasonal profile.")
    work["month"] = work[date_column].dt.month
    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
                  5: "Pre-monsoon", 6: "Southwest monsoon", 7: "Southwest monsoon",
                  8: "Southwest monsoon", 9: "Southwest monsoon", 10: "Post-monsoon",
                  11: "Post-monsoon"}
    work["season"] = work["month"].map(season_map)
    seasonal = work.groupby("season")[value_column].mean().reindex(SEASON_ORDER)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=seasonal.index, y=seasonal.values, marker_color="#ff7f0e"))
    fig.update_layout(
        title=title or f"Seasonal mean {value_column} (Kerala definition)",
        xaxis_title="Season",
        yaxis_title=f"Mean {value_column}",
        template="plotly_white",
        height=450,
    )
    return FigureResult(fig, f"{len(work)} observations.")


def missing_value_bar(df: pd.DataFrame) -> FigureResult:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Missing Values by Column",
            annotations=[{"text": "No missing values.", "xref": "paper", "yref": "paper", "showarrow": False}],
            template="plotly_white", height=400,
        )
        return FigureResult(fig, "No missing values.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=missing.index.tolist(), y=missing.values, marker_color="#d62728"))
    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="Column",
        yaxis_title="Missing count",
        template="plotly_white",
        height=450,
    )
    fig.update_xaxes(tickangle=45)
    return FigureResult(fig, f"{len(missing)} column(s) with missing values.")


def correlation_heatmap(df: pd.DataFrame, columns: Optional[list[str]] = None) -> FigureResult:
    numeric = df.select_dtypes(include=[np.number])
    if columns:
        numeric = numeric[[c for c in columns if c in numeric.columns]]
    if numeric.shape[1] < 2:
        return _empty_figure("Need at least two numeric columns for correlation.")
    corr = numeric.corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title="Correlation Matrix (numeric variables)",
        template="plotly_white",
        height=500,
    )
    return FigureResult(fig, f"{numeric.shape[1]} numeric variable(s).")


def aq_weather_scatter(
    df: pd.DataFrame,
    aq_column: str,
    wx_column: str,
    location_column: Optional[str] = None,
) -> FigureResult:
    ok, msg = _require(df, [aq_column, wx_column])
    if not ok:
        return _empty_figure(msg)
    work = df.dropna(subset=[aq_column, wx_column])
    if work.empty:
        return _empty_figure(f"No paired observations for {aq_column} vs {wx_column}.")
    fig = px.scatter(
        work, x=wx_column, y=aq_column, color=location_column if location_column in work.columns else None,
        title=f"{aq_column} vs {wx_column} (paired observations)",
        labels={aq_column: aq_column, wx_column: wx_column},
        template="plotly_white",
        height=450,
    )
    return FigureResult(fig, f"{len(work)} paired observations.")
