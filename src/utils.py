"""Utilidades compartidas por la aplicación."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


def is_numeric_col(series: pd.Series) -> bool:
    if ptypes.is_bool_dtype(series):
        return False
    return ptypes.is_numeric_dtype(series)


def is_date_col(series: pd.Series) -> bool:
    return ptypes.is_datetime64_any_dtype(series)


def is_text_col(series: pd.Series) -> bool:
    if is_numeric_col(series) or is_date_col(series) or ptypes.is_bool_dtype(series):
        return False
    return ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric_col(df[c])]


def date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_date_col(df[c])]


def text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_text_col(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int | None = None) -> list[str]:
    cols: list[str] = []
    for col in text_columns(df):
        n = df[col].nunique(dropna=True)
        if max_unique is not None and (n < 2 or n > max_unique):
            continue
        cols.append(col)
    return cols


def to_snake_case(name: str) -> str:
    """Convierte 'Ingreso Total ($)' en 'ingreso_total'."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^\w\s]", " ", text).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "columna"
    if text[0].isdigit():
        text = f"col_{text}"
    return text


def unique_column_names(columns: list[str]) -> tuple[list[str], dict[str, str]]:
    """Normaliza nombres y resuelve duplicados añadiendo sufijos."""
    result: list[str] = []
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for original in columns:
        base = to_snake_case(original)
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        used.add(candidate)
        result.append(candidate)
        mapping[str(original)] = candidate
    return result, mapping


def remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def fmt_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(value):
        return "—"

    def swap(text: str) -> str:
        return text.replace(",", "X").replace(".", ",").replace("X", ".")

    if abs(value) >= 1_000_000_000:
        return swap(f"{value / 1_000_000_000:,.{decimals}f}") + " mil M"
    if abs(value) >= 1_000_000:
        return swap(f"{value / 1_000_000:,.{decimals}f}") + " M"
    if abs(value) >= 1000:
        return swap(f"{value:,.{decimals}f}")
    return swap(f"{value:,.{decimals}f}")


def fmt_compact(value: float | int | None) -> str:
    """Formato corto para etiquetas de gráficas (1,75 M · 474,7 K)."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(value):
        return "—"
    if abs(value) >= 1_000_000:
        return fmt_number(value, 2)
    if abs(value) >= 10_000:
        return fmt_number(value / 1000, 1) + " K"
    return fmt_number(value, 0)


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{decimals}f}%".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def safe_filename(name: str, suffix: str) -> str:
    base = re.sub(r"[^\w\-. ]", "", str(name)).strip() or "dataset"
    base = re.sub(r"\s+", "_", base)
    if base.lower().endswith(suffix.lower()):
        return base
    return f"{base}{suffix}"


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}".replace(".", ",")
        size /= 1024
    return f"{size:.1f} TB".replace(".", ",")


def truncate(text: str, max_len: int = 40) -> str:
    text = str(text)
    return text if len(text) <= max_len else f"{text[: max_len - 1]}…"


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_python_scalar(value):
    """Convierte tipos de numpy/pandas a tipos nativos (para exportaciones)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    return value
