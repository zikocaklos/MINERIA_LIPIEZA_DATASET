"""Perfilado, diagnóstico de calidad y reseñas del dataset."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .cleaning import _DATE_LIKE_RE, _looks_like_id, numeric_ratio, parse_date, parse_number
from .config import PLACEHOLDER_NULLS, QUALITY_WEIGHTS
from .utils import (
    categorical_columns,
    date_columns,
    fmt_number,
    fmt_pct,
    is_date_col,
    is_missing,
    is_numeric_col,
    is_text_col,
    numeric_columns,
    remove_accents,
    text_columns,
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", remove_accents(str(value)).strip().lower())


def column_consistency(df: pd.DataFrame) -> float:
    """Proporción de columnas sin ruido de texto, placeholders ni tipos mixtos."""
    if df is None or df.empty or not len(df.columns):
        return 0.0
    healthy = 0
    for col in df.columns:
        series = df[col]
        if not is_text_col(series):
            healthy += 1
            continue
        values = series.dropna()
        if values.empty:
            healthy += 1
            continue
        unique = pd.unique(values)
        if len(unique) > 500:
            unique = unique[:500]
        text = pd.Series(unique, dtype="object").astype(str)
        normalized = text.map(_normalize_text)
        has_noise = normalized.nunique() < len(set(text))
        has_placeholder = bool(text.str.strip().str.lower().isin(PLACEHOLDER_NULLS).any())
        looks_numeric = numeric_ratio(text) >= 0.8
        looks_date = (
            sum(1 for value in text if not pd.isna(parse_date(value))) / len(text) >= 0.8
        )
        if has_noise or has_placeholder or looks_numeric or looks_date:
            continue
        healthy += 1
    return healthy / len(df.columns)


# ---------------------------------------------------------------------------
# Resumen y calidad
# ---------------------------------------------------------------------------
def quality_components(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "completitud": 0.0,
            "unicidad": 0.0,
            "columnas_utiles": 0.0,
            "consistencia": 0.0,
            "puntaje": 0.0,
        }
    total_cells = df.shape[0] * df.shape[1]
    completeness = 1 - (float(df.isna().sum().sum()) / total_cells) if total_cells else 0.0
    duplicates = float(df.duplicated().mean()) if len(df) else 0.0
    useful = 1.0
    if len(df.columns):
        empty = sum(1 for c in df.columns if df[c].notna().sum() == 0)
        constant = sum(1 for c in df.columns if df[c].nunique(dropna=False) <= 1)
        useful = 1 - ((empty + constant) / len(df.columns))
    consistency = column_consistency(df)
    score = 100 * (
        QUALITY_WEIGHTS["completitud"] * completeness
        + QUALITY_WEIGHTS["unicidad"] * (1 - duplicates)
        + QUALITY_WEIGHTS["columnas_utiles"] * useful
        + QUALITY_WEIGHTS["consistencia"] * consistency
    )
    return {
        "completitud": float(completeness),
        "unicidad": float(1 - duplicates),
        "columnas_utiles": float(useful),
        "consistencia": float(consistency),
        "puntaje": float(max(0.0, min(100.0, score))),
    }


def quality_score(df: pd.DataFrame) -> float:
    return quality_components(df)["puntaje"]


def quality_label(score: float) -> tuple[str, str]:
    if score >= 90:
        return "Excelente", "🟢"
    if score >= 75:
        return "Buena", "🟡"
    if score >= 55:
        return "Regular", "🟠"
    return "Baja", "🔴"


def dataset_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "filas": 0,
            "columnas": 0,
            "memoria_mb": 0.0,
            "nulos": 0,
            "nulos_pct": 0.0,
            "duplicados": 0,
            "duplicados_pct": 0.0,
            "numericas": 0,
            "categoricas": 0,
            "fechas": 0,
            "texto_libre": 0,
        }
    text_cols = text_columns(df)
    categorical = categorical_columns(df)
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "memoria_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        "nulos": int(df.isna().sum().sum()),
        "nulos_pct": float(df.isna().mean().mean()),
        "duplicados": int(df.duplicated().sum()),
        "duplicados_pct": float(df.duplicated().mean()),
        "numericas": len(numeric_columns(df)),
        "categoricas": len(categorical),
        "fechas": len(date_columns(df)),
        "texto_libre": len([c for c in text_cols if c not in categorical]),
    }


def missing_summary(df: pd.DataFrame, only_with_missing: bool = False) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        n = int(df[col].isna().sum())
        if only_with_missing and n == 0:
            continue
        rows.append(
            {
                "Columna": col,
                "Tipo": str(df[col].dtype),
                "Nulos": n,
                "% Nulos": n / len(df) if len(df) else 0,
                "Válidos": int(df[col].notna().sum()),
                "Únicos": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        info = {
            "Columna": col,
            "Tipo": _type_label(s),
            "No nulos": non_null,
            "% Nulos": float(s.isna().mean()) if n else 0.0,
            "Únicos": int(s.nunique(dropna=True)),
            "Ejemplo": _example(s),
        }
        if is_numeric_col(s) and non_null:
            valid = s.dropna()
            info.update(
                {
                    "Mínimo": float(valid.min()),
                    "Mediana": float(valid.median()),
                    "Media": float(valid.mean()),
                    "Máximo": float(valid.max()),
                }
            )
        elif is_date_col(s) and non_null:
            valid = s.dropna()
            info.update(
                {
                    "Mínimo": valid.min(),
                    "Mediana": valid.median(),
                    "Media": valid.mean(),
                    "Máximo": valid.max(),
                }
            )
        else:
            info.update({"Mínimo": None, "Mediana": None, "Media": None, "Máximo": None})
        rows.append(info)
    return pd.DataFrame(rows)


def _type_label(series: pd.Series) -> str:
    if is_numeric_col(series):
        return "Numérica"
    if is_date_col(series):
        return "Fecha"
    if is_text_col(series):
        return "Categórica" if series.nunique(dropna=True) <= 50 else "Texto"
    return str(series.dtype)


def _example(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "—"
    value = values.iloc[0]
    text = str(value)
    return text if len(text) <= 40 else f"{text[:39]}…"


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
def diagnose(df: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict] = []
    n = len(df)

    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(
            {
                "Severidad": "Alta" if duplicates / max(n, 1) > 0.05 else "Media",
                "Columna": "(todas)",
                "Problema": "Filas duplicadas",
                "Detalle": f"{duplicates:,} filas repetidas ({duplicates / n:.1%})".replace(",", "."),
                "Sugerencia": "Activa 'Eliminar duplicados' en la limpieza.",
            }
        )

    for col in df.columns:
        s = df[col]
        null_pct = float(s.isna().mean())
        if null_pct > 0:
            sev = "Alta" if null_pct >= 0.5 else ("Media" if null_pct >= 0.15 else "Baja")
            issues.append(
                {
                    "Severidad": sev,
                    "Columna": col,
                    "Problema": "Valores nulos",
                    "Detalle": f"{int(s.isna().sum()):,} nulos ({null_pct:.1%})".replace(",", "."),
                    "Sugerencia": "Imputar (mediana/moda) o eliminar si supera el 80%.",
                }
            )
        if s.nunique(dropna=False) <= 1:
            issues.append(
                {
                    "Severidad": "Media",
                    "Columna": col,
                    "Problema": "Columna constante",
                    "Detalle": "Todos los registros tienen el mismo valor (o está vacía).",
                    "Sugerencia": "Eliminarla: no aporta información al análisis.",
                }
            )
        if not is_text_col(s) or s.dropna().empty:
            continue

        values = s.dropna().astype(str)
        sample = values.head(5000)

        whitespace = int((sample != sample.str.strip()).sum())
        if whitespace:
            issues.append(
                {
                    "Severidad": "Baja",
                    "Columna": col,
                    "Problema": "Espacios sobrantes",
                    "Detalle": f"{whitespace:,} valores con espacios al inicio/final.".replace(",", "."),
                    "Sugerencia": "Activa la limpieza de texto.",
                }
            )

        placeholders = int(sample.str.strip().str.lower().isin(PLACEHOLDER_NULLS).sum())
        if placeholders:
            issues.append(
                {
                    "Severidad": "Media",
                    "Columna": col,
                    "Problema": "Nulos encubiertos",
                    "Detalle": f"{placeholders:,} valores tipo 'N/A', '-', 'sin dato'.".replace(",", "."),
                    "Sugerencia": "Conviértelos a nulos reales (opción activa por defecto).",
                }
            )

        stripped = sample.str.strip()
        distinct_spellings = stripped.groupby(stripped.str.lower()).nunique()
        variant_groups = int((distinct_spellings > 1).sum())
        if variant_groups:
            issues.append(
                {
                    "Severidad": "Baja",
                    "Columna": col,
                    "Problema": "Capitalización inconsistente",
                    "Detalle": f"{variant_groups:,} valores aparecen escritos con mayúsculas y minúsculas distintas.".replace(",", "."),
                    "Sugerencia": "Elige 'Tipo Título' o 'minúsculas' en el formato de texto.",
                }
            )

        if s.nunique(dropna=True) > max(20, 0.5 * len(s)) and not _looks_like_id(col, s):
            issues.append(
                {
                    "Severidad": "Baja",
                    "Columna": col,
                    "Problema": "Alta cardinalidad",
                    "Detalle": f"{s.nunique(dropna=True):,} valores distintos; podría ser un identificador.".replace(",", "."),
                    "Sugerencia": "Consérvala como texto; no la uses como categoría en gráficas.",
                }
            )

        types = sample.map(lambda v: type(v).__name__).nunique()
        if types > 1:
            issues.append(
                {
                    "Severidad": "Media",
                    "Columna": col,
                    "Problema": "Tipos mezclados",
                    "Detalle": f"Se detectaron {types} tipos de dato en la misma columna.",
                    "Sugerencia": "Revisa los valores atípicos y fuerza un tipo en la limpieza.",
                }
            )

        if _looks_like_id(col, s):
            continue

        if numeric_ratio(s) >= 0.85:
            issues.append(
                {
                    "Severidad": "Media",
                    "Columna": col,
                    "Problema": "Números guardados como texto",
                    "Detalle": "La mayoría de los valores son numéricos pero el tipo es texto.",
                    "Sugerencia": "Activa la conversión automática de tipos.",
                }
            )
        elif _DATE_LIKE_RE.search(" ".join(sample.head(30).tolist())):
            from .cleaning import date_ratio

            if date_ratio(s) >= 0.8:
                issues.append(
                    {
                        "Severidad": "Media",
                        "Columna": col,
                        "Problema": "Fechas guardadas como texto",
                        "Detalle": "Los valores parecen fechas pero están como texto.",
                        "Sugerencia": "Activa la conversión automática de fechas.",
                    }
                )

    if not issues:
        issues.append(
            {
                "Severidad": "Baja",
                "Columna": "—",
                "Problema": "Sin problemas evidentes",
                "Detalle": "No se detectaron nulos, duplicados ni inconsistencias.",
                "Sugerencia": "El dataset está listo para analizar.",
            }
        )

    order = {"Alta": 0, "Media": 1, "Baja": 2}
    result = pd.DataFrame(issues)
    result["_orden"] = result["Severidad"].map(order)
    result = result.sort_values(["_orden", "Columna"]).drop(columns="_orden").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Roles y valores frecuentes
# ---------------------------------------------------------------------------
def column_roles(df: pd.DataFrame) -> dict:
    numeric = numeric_columns(df)
    dates = date_columns(df)
    categorical = [
        c
        for c in categorical_columns(df, max_unique=60)
        if df[c].nunique(dropna=True) >= 2
    ]
    free_text = [c for c in text_columns(df) if c not in categorical]
    ids = [c for c in text_columns(df) if _looks_like_id(c, df[c])]
    return {
        "numeric": numeric,
        "date": dates,
        "categorical": categorical,
        "text": free_text,
        "ids": ids,
    }


def top_values(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False).head(n)
    result = counts.rename_axis(column).reset_index(name="Conteo")
    result["% del total"] = result["Conteo"] / max(len(df), 1)
    result[column] = result[column].astype(str).replace({"<NA>": "(nulo)", "nan": "(nulo)"})
    return result


def numeric_summary(df: pd.DataFrame, column: str) -> dict:
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return {}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "q1": float(q1),
        "q3": float(q3),
        "max": float(s.max()),
        "sum": float(s.sum()),
        "skew": float(s.skew()) if len(s) > 2 else 0.0,
    }


def detect_business_metrics(df: pd.DataFrame) -> dict[str, str]:
    """Sugiere columnas de métrica, fecha y dimensión para el dashboard."""
    roles = column_roles(df)
    priority_groups = (
        ("ingreso", "venta", "monto", "revenue", "sales", "amount", "facturacion", "facturación"),
        ("margen", "utilidad", "ganancia", "profit"),
        ("precio", "price", "tarifa"),
        ("costo", "cost", "gasto", "expense"),
        ("unidades", "cantidad", "qty", "quantity"),
    )
    metric = None
    for group in priority_groups:
        for col in roles["numeric"]:
            if any(key in col.lower() for key in group):
                metric = col
                break
        if metric:
            break
    if metric is None and roles["numeric"]:
        metric = roles["numeric"][0]

    date_keywords = ("fecha", "date", "periodo", "mes", "dia", "year", "anio")
    geo_keywords = ("region", "ciudad", "pais", "zona", "sucursal", "tienda", "canal", "segmento")

    date_col = next(
        (c for c in roles["date"] if any(k in c.lower() for k in date_keywords)),
        roles["date"][0] if roles["date"] else None,
    )
    dimension = next(
        (c for c in roles["categorical"] if any(k in c.lower() for k in geo_keywords)),
        roles["categorical"][0] if roles["categorical"] else None,
    )
    return {"metric": metric, "date": date_col, "dimension": dimension}


def format_profile_table(profile: pd.DataFrame) -> pd.DataFrame:
    """Prepara el perfil para mostrarse en la interfaz."""
    out = profile.copy()
    for col in ["% Nulos"]:
        if col in out:
            out[col] = out[col].map(lambda v: fmt_pct(v))
    for col in ["Mínimo", "Mediana", "Media", "Máximo"]:
        if col in out:
            out[col] = out[col].map(
                lambda v: fmt_number(v) if isinstance(v, (int, float, np.number)) and not pd.isna(v) else "—"
            )
    return out
