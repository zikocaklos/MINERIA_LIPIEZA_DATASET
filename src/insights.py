"""Hallazgos automáticos y recomendaciones para la toma de decisiones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PALETTE  # noqa: F401  (se usa en informes)
from .profiling import detect_business_metrics, quality_components
from .utils import (
    categorical_columns,
    date_columns,
    fmt_number,
    fmt_pct,
    is_numeric_col,
    numeric_columns,
)


def _insight(level: str, title: str, detail: str, action: str = "", kind: str = "general") -> dict:
    return {
        "nivel": level,
        "titulo": title,
        "detalle": detail,
        "accion": action,
        "tipo": kind,
    }


def _numeric_outliers(df: pd.DataFrame, limit: int = 8) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    for col in numeric_columns(df)[:limit]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < low) | (s > high)).sum())
        if count:
            rows.append((col, count, count / len(s)))
    return sorted(rows, key=lambda x: x[2], reverse=True)


def generate_insights(
    df: pd.DataFrame,
    stats: dict | None = None,
    max_items: int = 10,
) -> list[dict]:
    """Genera hallazgos legibles a partir del dataset ya limpio."""
    if df is None or df.empty:
        return []

    stats = stats or {}
    insights: list[dict] = []
    roles = detect_business_metrics(df)
    metric, date_col, dimension = roles["metric"], roles["date"], roles["dimension"]
    numeric = numeric_columns(df)
    cats = categorical_columns(df, max_unique=50)

    quality = quality_components(df)
    score = quality["puntaje"]
    if score >= 90:
        insights.append(
            _insight(
                "success",
                f"Calidad de datos: {score:.0f}/100",
                f"El dataset quedó con {fmt_pct(quality['completitud'])} de celdas completas "
                f"y {fmt_pct(1 - quality['unicidad'])} de filas duplicadas.",
                "Publica el tablero con confianza y programa una actualización periódica.",
                "calidad",
            )
        )
    else:
        insights.append(
            _insight(
                "warning",
                f"Calidad de datos: {score:.0f}/100",
                f"Queda {fmt_pct(1 - quality['completitud'])} de celdas vacías. "
                "Revisa las columnas con más nulos antes de concluir.",
                "Refuerza la captura de datos en los campos críticos.",
                "calidad",
            )
        )

    removed = int(stats.get("filas_eliminadas", 0))
    duplicates = int(stats.get("duplicados_eliminados", 0))
    if duplicates:
        insights.append(
            _insight(
                "info",
                f"{fmt_number(duplicates, 0)} registros duplicados eliminados",
                f"Se depuraron {fmt_number(removed, 0)} filas en total "
                f"({fmt_pct(removed / max(stats.get('filas_antes', len(df) + removed), 1))} del original).",
                "Revisa por qué se generan duplicados en el sistema de origen.",
                "limpieza",
            )
        )

    # --- Concentración / Pareto -------------------------------------------------
    if dimension and metric and dimension in df.columns and metric in df.columns:
        grouped = (
            df.dropna(subset=[dimension])
            .groupby(dimension, observed=True)[metric]
            .sum()
            .sort_values(ascending=False)
        )
        grouped = grouped[grouped > 0]
        if len(grouped) >= 3:
            total = grouped.sum()
            share = grouped / total
            top_name = str(grouped.index[0])
            top_share = float(share.iloc[0])
            cumulative = share.cumsum()
            n80 = int((cumulative < 0.8).sum() + 1)
            detail = (
                f"'{top_name}' concentra {fmt_pct(top_share)} de {metric.replace('_', ' ')}. "
                f"Se necesitan {n80} de {len(grouped)} categorías para llegar al 80%."
            )
            if top_share >= 0.4:
                action = (
                    f"Dependencia alta: diseña un plan de crecimiento para las categorías "
                    f"secundarias y protege la principal."
                )
                level = "warning"
            else:
                action = "La base está diversificada; prioriza las categorías con mayor margen."
                level = "success" if n80 > 3 else "info"
            insights.append(
                _insight(
                    level,
                    f"Concentración por {dimension.replace('_', ' ')}",
                    detail,
                    action,
                    "concentracion",
                )
            )

        if len(grouped) >= 2:
            top2 = grouped.head(2)
            insights.append(
                _insight(
                    "info",
                    f"Mayor aporte: {top_name}",
                    f"Genera {fmt_number(float(top2.iloc[0]))} de {metric.replace('_', ' ')} "
                    f"({fmt_pct(top_share)} del total).",
                    f"Analiza qué está funcionando en '{top_name}' y replica las prácticas.",
                    "ranking",
                )
            )

    # --- Tendencia temporal ------------------------------------------------------
    if date_col and metric and date_col in df.columns and metric in df.columns:
        work = df.dropna(subset=[date_col]).copy()
        if len(work) >= 6 and is_numeric_col(work[metric]):
            try:
                monthly = (
                    work.set_index(date_col)[metric]
                    .resample("MS")
                    .sum()
                    .dropna()
                )
            except (TypeError, ValueError):
                monthly = pd.Series(dtype=float)
            if len(monthly) >= 4:
                recent = monthly.tail(3).mean()
                previous = monthly.iloc[-6:-3].mean() if len(monthly) >= 6 else monthly.iloc[:-3].mean()
                if previous and not pd.isna(previous) and previous != 0:
                    change = (recent - previous) / abs(previous)
                    direction = "creció" if change >= 0 else "cayó"
                    level = "success" if change >= 0.05 else ("danger" if change <= -0.05 else "info")
                    insights.append(
                        _insight(
                            level,
                            f"Tendencia: {direction} {fmt_pct(abs(change))}",
                            f"El promedio de los últimos 3 meses "
                            f"({fmt_number(float(recent))}) se compara con los 3 anteriores "
                            f"({fmt_number(float(previous))}).",
                            "Ajusta inventario, presupuesto y metas de venta a esta tendencia.",
                            "tendencia",
                        )
                    )
                best_month = monthly.idxmax()
                best_value = float(monthly.max())
                insights.append(
                    _insight(
                        "info",
                        f"Mejor periodo: {best_month:%m/%Y}",
                        f"Alcanzó {fmt_number(best_value)} de {metric.replace('_', ' ')}.",
                        "Identifica la causa (campaña, estacionalidad, precio) y prepárate para repetirla.",
                        "tendencia",
                    )
                )

    # --- Correlaciones -----------------------------------------------------------
    if len(numeric) >= 2:
        corr = df[numeric].apply(pd.to_numeric, errors="coerce").corr()
        pairs = []
        for i, a in enumerate(numeric):
            for b in numeric[i + 1 :]:
                value = corr.loc[a, b]
                if pd.notna(value) and abs(value) >= 0.65:
                    pairs.append((a, b, float(value)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for a, b, value in pairs[:2]:
            relation = "positiva" if value > 0 else "negativa"
            insights.append(
                _insight(
                    "info",
                    f"Relación {relation} entre {a.replace('_', ' ')} y {b.replace('_', ' ')}",
                    f"Coeficiente de correlación r = {value:.2f}. "
                    + (
                        "Cuando una sube, la otra también tiende a subir."
                        if value > 0
                        else "Cuando una sube, la otra tiende a bajar."
                    ),
                    "Úsalo para priorizar palancas, pero valida con un análisis causal antes de decidir.",
                    "correlacion",
                )
            )

    # --- Nulos residuales --------------------------------------------------------
    missing = df.isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        top_col = str(missing.index[0])
        insights.append(
            _insight(
                "warning",
                f"Columna con más vacíos: {top_col}",
                f"El {fmt_pct(missing.iloc[0])} de sus valores sigue vacío tras la limpieza.",
                "Decide si la columna es indispensable; si no, exclúyela de los tableros.",
                "calidad",
            )
        )

    # --- Atípicos ----------------------------------------------------------------
    outliers = _numeric_outliers(df)
    if outliers:
        col, count, share = outliers[0]
        insights.append(
            _insight(
                "warning" if share > 0.05 else "info",
                f"Valores atípicos en {col.replace('_', ' ')}",
                f"{fmt_number(count, 0)} registros ({fmt_pct(share)}) están fuera del rango "
                "habitual (regla 1,5 × IQR).",
                "Verifica si son errores de captura o casos reales de alto impacto.",
                "atipicos",
            )
        )

    # --- Dispersión / asimetría -------------------------------------------------
    if numeric:
        skews = []
        for col in numeric[:10]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 10:
                skews.append((col, float(s.skew())))
        extreme = [x for x in skews if abs(x[1]) >= 1.5]
        if extreme:
            col, skew = extreme[0]
            insights.append(
                _insight(
                    "info",
                    f"Distribución sesgada en {col.replace('_', ' ')}",
                    f"Asimetría de {skew:.2f}: hay muchos valores bajos y pocos valores muy altos "
                    "(o al revés).",
                    "Usa la mediana en lugar del promedio para describir esta variable.",
                    "distribucion",
                )
            )

    # --- Diversidad de categorías ------------------------------------------------
    if not dimension and cats:
        col = cats[0]
        counts = df[col].astype(str).value_counts()
        if len(counts) >= 2:
            insights.append(
                _insight(
                    "info",
                    f"Categoría dominante en {col.replace('_', ' ')}",
                    f"'{counts.index[0]}' representa {fmt_pct(counts.iloc[0] / len(df))} de los registros.",
                    "Revisa si la muestra está equilibrada antes de comparar segmentos.",
                    "categorias",
                )
            )

    if not insights:
        insights.append(
            _insight(
                "info",
                "Datos listos para explorar",
                "No se detectaron patrones automáticos relevantes. Usa el constructor de gráficas "
                "para plantear tus propias preguntas de negocio.",
                "Define la métrica y la dimensión clave de tu área.",
                "general",
            )
        )

    return insights[:max_items]


def executive_summary(df: pd.DataFrame, stats: dict | None = None) -> str:
    """Resumen ejecutivo en texto para el informe."""
    stats = stats or {}
    roles = detect_business_metrics(df)
    quality = quality_components(df)
    parts = [
        f"El dataset contiene {len(df):,} registros y {df.shape[1]} columnas "
        f"tras aplicar la limpieza."
    ]
    if stats.get("filas_eliminadas"):
        parts.append(
            f"Se eliminaron {stats['filas_eliminadas']:,} filas y "
            f"{stats.get('columnas_eliminadas', 0)} columnas."
        )
    parts.append(
        f"La completitud final es de {quality['completitud']:.1%} y el puntaje de calidad "
        f"es {quality['puntaje']:.0f}/100."
    )
    if roles.get("metric"):
        metric = roles["metric"].replace("_", " ")
        parts.append(f"La métrica principal sugerida es '{metric}'")
        if roles.get("dimension"):
            parts.append(f"analizada por '{roles['dimension'].replace('_', ' ')}'")
        if roles.get("date"):
            parts.append(f"a lo largo de '{roles['date'].replace('_', ' ')}'")
        parts[-1] = parts[-1].rstrip(".") + "."
    return " ".join(parts).replace("  ", " ")
