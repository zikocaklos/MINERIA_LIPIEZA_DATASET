"""Gráficas reutilizables (Plotly) para el análisis y la toma de decisiones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CHART_TEMPLATE, PALETTE
from .utils import (
    categorical_columns,
    date_columns,
    fmt_compact,
    is_date_col,
    is_numeric_col,
    numeric_columns,
)


def pretty(name: str) -> str:
    """Nombre legible para ejes y títulos."""
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text


def _style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        colorway=PALETTE,
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#0F172A"),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16), x=0.01, xanchor="left"))
    fig.update_xaxes(showgrid=False, title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#E2E8F0", title_font=dict(size=12), zeroline=False)
    return fig


def empty_figure(message: str = "No hay datos suficientes para esta gráfica.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#64748B"),
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=260,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _sample(df: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=42)


# ---------------------------------------------------------------------------
# Calidad y distribuciones
# ---------------------------------------------------------------------------
def missing_values_chart(df: pd.DataFrame, height: int = 380) -> go.Figure:
    if df.empty:
        return empty_figure()
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        fig = empty_figure("🎉 No hay valores nulos en el dataset.")
        return fig
    data = missing.reset_index()
    data.columns = ["Columna", "Nulos"]
    data["Etiqueta"] = data["Nulos"].map(lambda v: f"{v:.1%}")
    fig = px.bar(
        data,
        x="Nulos",
        y="Columna",
        orientation="h",
        text="Etiqueta",
        color="Nulos",
        color_continuous_scale=["#DBEAFE", "#2563EB"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
    fig.update_xaxes(tickformat=".0%", range=[0, min(1, missing.max() * 1.25)])
    return _style(fig, "Porcentaje de valores nulos por columna", height)


def numeric_distribution(df: pd.DataFrame, column: str, bins: int = 40) -> go.Figure:
    if column not in df or not is_numeric_col(df[column]):
        return empty_figure()
    data = df[[column]].dropna()
    if data.empty:
        return empty_figure()
    bins = int(max(10, min(bins, max(10, data[column].nunique()))))
    fig = px.histogram(
        data,
        x=column,
        nbins=bins,
        marginal="box",
        color_discrete_sequence=[PALETTE[0]],
        labels={column: pretty(column)},
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _style(fig, f"Distribución de {pretty(column)}", 420)


def numeric_grid(df: pd.DataFrame, columns: list[str], max_cols: int = 3) -> go.Figure:
    columns = [c for c in columns if c in df.columns and is_numeric_col(df[c])][:6]
    if not columns:
        return empty_figure()
    rows = int(np.ceil(len(columns) / max_cols))
    fig = make_subplots(
        rows=rows,
        cols=max_cols,
        subplot_titles=[pretty(c) for c in columns],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )
    for i, col in enumerate(columns):
        r, c = divmod(i, max_cols)
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=30,
                marker_color=PALETTE[i % len(PALETTE)],
                opacity=0.9,
                name=pretty(col),
                showlegend=False,
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(bargap=0.05)
    return _style(fig, "Distribución de las principales variables numéricas", 260 * rows + 60)


def correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None) -> go.Figure:
    cols = [
        c
        for c in (columns or numeric_columns(df))
        if c in df.columns and is_numeric_col(df[c]) and df[c].nunique(dropna=True) > 1
    ]
    if len(cols) < 2:
        return empty_figure("Se necesitan al menos dos columnas numéricas con variación.")
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    labels = [pretty(c) for c in cols]
    fig = px.imshow(
        corr,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_traces(textfont_size=10)
    fig.update_layout(coloraxis_colorbar=dict(title="r", thickness=12))
    height = max(360, 45 * len(cols) + 120)
    return _style(fig, "Matriz de correlación (Pearson)", height)


# ---------------------------------------------------------------------------
# Categorías
# ---------------------------------------------------------------------------
def category_bar(
    df: pd.DataFrame,
    column: str,
    value: str | None = None,
    agg: str = "sum",
    top_n: int = 15,
) -> go.Figure:
    if column not in df.columns:
        return empty_figure()
    work = df.copy()
    if value and value in work.columns and is_numeric_col(work[value]):
        grouped = (
            work.dropna(subset=[column])
            .groupby(column, observed=True)[value]
            .agg(agg)
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        metric_label = f"{agg} de {pretty(value)}"
    else:
        grouped = (
            work[column]
            .astype(str)
            .value_counts()
            .head(top_n)
            .rename_axis(column)
            .reset_index(name="Registros")
        )
        metric_label = "Registros"
    if grouped.empty:
        return empty_figure()
    grouped[column] = grouped[column].astype(str)
    value_col = grouped.columns[-1]
    grouped = grouped.sort_values(value_col, ascending=True)
    fig = px.bar(
        grouped,
        x=value_col,
        y=column,
        orientation="h",
        text=[fmt_compact(v) for v in grouped[value_col]],
        color=value_col,
        color_continuous_scale=["#DBEAFE", "#1D4ED8"],
        labels={column: pretty(column), value_col: metric_label},
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=metric_label)
    fig = _style(fig, f"{pretty(column)} · top {min(top_n, len(grouped))}", 420)
    fig.update_layout(margin=dict(l=8, r=64, t=48, b=8))
    return fig


def pareto_chart(
    df: pd.DataFrame,
    column: str,
    value: str | None = None,
    agg: str = "sum",
    top_n: int = 12,
) -> go.Figure:
    if column not in df.columns:
        return empty_figure()
    work = df.dropna(subset=[column]).copy()
    if work.empty:
        return empty_figure()
    if value and value in work.columns and is_numeric_col(work[value]):
        grouped = work.groupby(column, observed=True)[value].agg(agg).sort_values(ascending=False)
        label = f"{agg} de {pretty(value)}"
    else:
        grouped = work[column].astype(str).value_counts()
        label = "Registros"
    if grouped.empty:
        return empty_figure()
    total = grouped.sum()
    grouped = grouped.head(top_n)
    cumulative = grouped.cumsum() / total if total else grouped * 0
    x = [str(v) for v in grouped.index]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=x,
            y=grouped.values,
            name=label,
            marker_color=PALETTE[0],
            text=[f"{v:,.0f}" for v in grouped.values],
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cumulative.values * 100,
            name="% acumulado",
            mode="lines+markers",
            line=dict(color=PALETTE[2], width=2.5),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="#94A3B8",
        secondary_y=True,
        annotation_text="80%",
        annotation_position="top left",
    )
    fig.update_yaxes(title_text=label, secondary_y=False)
    fig.update_yaxes(title_text="% acumulado", range=[0, 105], secondary_y=True, showgrid=False)
    fig.update_xaxes(tickangle=-25)
    return _style(fig, f"Pareto de {pretty(column)}", 440)


def donut_chart(df: pd.DataFrame, column: str, top_n: int = 8) -> go.Figure:
    if column not in df.columns:
        return empty_figure()
    counts = df[column].astype(str).value_counts(dropna=False)
    if counts.empty:
        return empty_figure()
    if len(counts) > top_n:
        head = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        head = pd.concat([head, pd.Series({"Otros": other})])
        counts = head
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.55,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textposition="inside", textinfo="percent", pull=[0.02] * len(counts))
    fig = _style(fig, f"Composición de {pretty(column)}", 400)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=8, r=8, t=48, b=48),
    )
    return fig


def treemap_chart(
    df: pd.DataFrame,
    path: list[str],
    value: str | None = None,
) -> go.Figure:
    path = [c for c in path if c in df.columns]
    if not path:
        return empty_figure()
    work = df.dropna(subset=path).copy()
    if work.empty:
        return empty_figure()
    for col in path:
        work[col] = work[col].astype(str)
    if value and value in work.columns and is_numeric_col(work[value]):
        grouped = work.groupby(path, observed=True)[value].sum().reset_index()
        fig = px.treemap(grouped, path=path, values=value, color=value,
                         color_continuous_scale="Blues")
    else:
        fig = px.treemap(work, path=path, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="label+percent parent")
    return _style(fig, "Composición jerárquica", 440)


# ---------------------------------------------------------------------------
# Relaciones y tiempo
# ---------------------------------------------------------------------------
def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    trendline: bool = True,
    max_points: int = 5000,
) -> go.Figure:
    if x not in df or y not in df:
        return empty_figure()
    work = df.dropna(subset=[x, y]).copy()
    if work.empty or not is_numeric_col(work[x]) or not is_numeric_col(work[y]):
        return empty_figure()
    work = _sample(work, max_points)
    fig = px.scatter(
        work,
        x=x,
        y=y,
        color=color if color in work.columns else None,
        opacity=0.65,
        labels={x: pretty(x), y: pretty(y), color: pretty(color or "")},
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    if trendline and work[x].nunique() > 2:
        try:
            coef = np.polyfit(pd.to_numeric(work[x]), pd.to_numeric(work[y]), 1)
            xs = np.linspace(work[x].min(), work[x].max(), 100)
            ys = coef[0] * xs + coef[1]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="Tendencia",
                    line=dict(color="#EF4444", width=2, dash="dash"),
                )
            )
        except (ValueError, TypeError, np.linalg.LinAlgError):
            pass
    return _style(fig, f"{pretty(y)} vs. {pretty(x)}", 430)


def box_by_category(
    df: pd.DataFrame,
    category: str,
    numeric: str,
    top_n: int = 10,
) -> go.Figure:
    if category not in df or numeric not in df:
        return empty_figure()
    work = df.dropna(subset=[category, numeric])
    if work.empty or not is_numeric_col(work[numeric]):
        return empty_figure()
    top = work[category].astype(str).value_counts().head(top_n).index
    work = work[work[category].astype(str).isin(top)].copy()
    work[category] = work[category].astype(str)
    fig = px.box(
        work,
        x=category,
        y=numeric,
        color=category,
        points="outliers",
        labels={category: pretty(category), numeric: pretty(numeric)},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-25)
    return _style(fig, f"{pretty(numeric)} por {pretty(category)}", 430)


def time_series(
    df: pd.DataFrame,
    date_col: str,
    value: str | None = None,
    freq: str = "MS",
    agg: str = "sum",
    color: str | None = None,
) -> go.Figure:
    if date_col not in df.columns or not is_date_col(df[date_col]):
        return empty_figure()
    work = df.dropna(subset=[date_col]).copy()
    if work.empty:
        return empty_figure()
    work["_periodo"] = work[date_col].dt.to_period(
        {"D": "D", "W": "W", "MS": "M", "QS": "Q", "YS": "Y"}.get(freq, "M")
    ).dt.to_timestamp()
    use_color = color if color in work.columns else None

    if value and value in work.columns and is_numeric_col(work[value]):
        if agg == "count":
            grouped = work.groupby(["_periodo"] + ([use_color] if use_color else []), observed=True).size().reset_index(name="valor")
            label = "Registros"
        else:
            grouped = (
                work.groupby(["_periodo"] + ([use_color] if use_color else []), observed=True)[value]
                .agg(agg)
                .reset_index(name="valor")
            )
            label = f"{agg} de {pretty(value)}"
    else:
        grouped = work.groupby(["_periodo"] + ([use_color] if use_color else []), observed=True).size().reset_index(name="valor")
        label = "Registros"

    if grouped.empty:
        return empty_figure()
    if use_color:
        grouped[use_color] = grouped[use_color].astype(str)
    fig = px.line(
        grouped,
        x="_periodo",
        y="valor",
        color=use_color,
        markers=True,
        labels={"_periodo": "Periodo", "valor": label, use_color: pretty(color or "")},
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(line=dict(width=2.5))
    return _style(fig, f"Evolución de {label}", 420)


# ---------------------------------------------------------------------------
# Constructor personalizado
# ---------------------------------------------------------------------------
def custom_chart(
    df: pd.DataFrame,
    kind: str,
    x: str,
    y: str | None = None,
    color: str | None = None,
    agg: str = "sum",
    top_n: int = 15,
    bins: int = 30,
) -> go.Figure:
    if x not in df.columns:
        return empty_figure()
    work = df.copy()
    color = color if color in work.columns else None

    if kind == "Histograma":
        if not is_numeric_col(work[x]):
            return empty_figure("Selecciona una columna numérica para el histograma.")
        fig = px.histogram(
            work.dropna(subset=[x]),
            x=x,
            nbins=bins,
            color=color,
            color_discrete_sequence=PALETTE,
            labels={x: pretty(x)},
        )
        return _style(fig, f"Distribución de {pretty(x)}", 420)

    if kind in {"Barras", "Barras horizontales"}:
        horizontal = kind == "Barras horizontales"
        has_value = bool(y and y in work.columns and is_numeric_col(work[y]))
        group_cols = [x] + ([color] if color and color != x else [])
        try:
            if has_value:
                grouped = (
                    work.dropna(subset=[x])
                    .groupby(group_cols, observed=True)[y]
                    .agg(agg)
                    .reset_index()
                    .sort_values(y, ascending=False)
                    .head(top_n)
                )
                value_col = y
                value_label = f"{agg} de {pretty(y)}"
            else:
                grouped = (
                    work.dropna(subset=[x])
                    .groupby(group_cols, observed=True)
                    .size()
                    .reset_index(name="Registros")
                    .sort_values("Registros", ascending=False)
                    .head(top_n)
                )
                value_col = "Registros"
                value_label = "Registros"
        except (ValueError, TypeError):
            return empty_figure("No se pudo agrupar con las columnas seleccionadas.")
        if grouped.empty:
            return empty_figure()
        grouped[x] = grouped[x].astype(str)
        color_arg = color if color else value_col
        common = dict(
            color=color_arg,
            labels={x: pretty(x), value_col: value_label},
        )
        if color:
            common["color_discrete_sequence"] = PALETTE
        else:
            common["color_continuous_scale"] = "Blues"
        if horizontal:
            grouped = grouped.sort_values(value_col)
            fig = px.bar(grouped, x=value_col, y=x, orientation="h", text_auto=True, **common)
            fig.update_layout(coloraxis_showscale=False)
        else:
            fig = px.bar(grouped, x=x, y=value_col, text_auto=True, **common)
            fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-25)
        return _style(fig, f"{pretty(x)} · {value_label}", 430)

    if kind == "Líneas":
        if not is_date_col(work[x]):
            return empty_figure("Selecciona una columna de fecha para el eje X.")
        return time_series(work, x, y if y and is_numeric_col(work[y]) else None,
                           freq="MS", agg=agg, color=color)

    if kind == "Área":
        fig = time_series(
            work,
            x,
            y if y and is_numeric_col(work[y]) else None,
            freq="MS",
            agg=agg,
            color=color,
        )
        fig.update_traces(fill="tozeroy")
        return fig

    if kind == "Dispersión":
        if not y or not is_numeric_col(work[x]) or not is_numeric_col(work[y]):
            return empty_figure("Selecciona dos columnas numéricas.")
        return scatter_plot(work, x, y, color=color)

    if kind == "Cajas":
        if not y:
            return empty_figure("Selecciona una métrica numérica.")
        return box_by_category(work, x, y, top_n=top_n)

    if kind in {"Torta", "Dona"}:
        return donut_chart(work, x, top_n=top_n)

    return empty_figure()
