"""Motor de limpieza de datos configurable y trazable."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil import parser as date_parser

from .config import DATE_HINTS, ID_HINTS, PLACEHOLDER_NULLS
from .utils import (
    date_columns,
    is_date_col,
    is_missing,
    is_numeric_col,
    is_text_col,
    numeric_columns,
    remove_accents,
    text_columns,
    to_snake_case,
    unique_column_names,
)

_CURRENCY_SYMBOLS = "$€£¥"
_CURRENCY_WORDS = (
    "usd",
    "cop",
    "mxn",
    "ars",
    "clp",
    "pen",
    "eur",
    "dolares",
    "dólares",
    "pesos",
    "euros",
    "soles",
)
_DATE_LIKE_RE = re.compile(
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{8}"
    r"|\d{4}[-/]\d{1,2}"
    r"|\d{1,2}\s+de\s+[a-záéíóúñ]+\s+de\s+\d{4}",
    re.IGNORECASE,
)
_ES_MONTHS = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "setiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}
_SENSITIVE_COLUMN_RE = re.compile(
    r"password|passwd|passphrase|pwd|contrasena|token|secret|credential|"
    r"credencial|authorization|api_?key|access_?key|private_?key|auth_?key|"
    r"(?:^|_)clave(?:_|$)|(?:^|_)firma(?:_|$)|signature",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass
class CleaningConfig:
    # Estructura
    normalize_columns: bool = True
    drop_empty_rows: bool = True
    drop_empty_cols: bool = True
    drop_constant_cols: bool = True
    drop_zero_cols: bool = True
    zero_dominance_threshold: float = 0.9
    drop_sensitive_cols: bool = True
    sensitive_cols: list[str] | None = None
    drop_high_null_cols: bool = False
    high_null_threshold: float = 0.8

    # Texto
    clean_text: bool = True
    remove_accents: bool = False
    case_mode: str = "none"  # none | lower | upper | title
    replace_placeholders: bool = True

    # Tipos
    auto_types: bool = True
    decimal_mode: str = "auto"  # auto | dot | comma
    dayfirst: bool = True
    force_numeric_cols: list[str] = field(default_factory=list)
    skip_numeric_cols: list[str] = field(default_factory=list)
    force_date_cols: list[str] = field(default_factory=list)
    skip_date_cols: list[str] = field(default_factory=list)

    # Duplicados
    drop_duplicates: bool = True
    duplicate_subset: list[str] | None = None

    # Nulos
    numeric_strategy: str = "median"  # median | mean | zero | ffill | drop_rows | drop_cols | none
    categorical_strategy: str = "mode"  # mode | constant | ffill | drop_rows | drop_cols | none
    constant_value: str = "Desconocido"
    impute_exclude: list[str] = field(default_factory=list)

    # Atípicos
    outlier_method: str = "iqr"  # iqr | zscore | none
    outlier_action: str = "clip"  # clip | remove | flag
    outlier_factor: float = 1.5
    outlier_cols: list[str] = field(default_factory=list)


@dataclass
class CleaningResult:
    df: pd.DataFrame
    log: pd.DataFrame
    stats: dict
    column_map: dict[str, str] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def removed_rows(self) -> int:
        return int(self.stats.get("filas_eliminadas", 0))


LOG_COLUMNS = ["Paso", "Detalle", "Filas afectadas", "Columnas afectadas"]


def _log(log: list[dict], step: str, detail: str, rows: int = 0, cols: int = 0) -> None:
    log.append(
        {
            "Paso": step,
            "Detalle": detail,
            "Filas afectadas": int(rows),
            "Columnas afectadas": int(cols),
        }
    )


def detect_sensitive_columns(df: pd.DataFrame) -> list[str]:
    """Detecta columnas de credenciales por el nombre, sin inspeccionar sus valores."""
    return [
        str(column)
        for column in df.columns
        if _SENSITIVE_COLUMN_RE.search(to_snake_case(str(column)))
    ]


def _zero_only_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        values = df[column].dropna()
        if values.empty:
            continue
        parsed = values.map(parse_number)
        if parsed.notna().all() and parsed.eq(0).all():
            columns.append(column)
    return columns


def _zero_dominant_binary_columns(df: pd.DataFrame, threshold: float) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        values = df[column].dropna()
        if values.empty:
            continue
        parsed = values.map(parse_number)
        if parsed.isna().any():
            continue
        unique = set(parsed.unique())
        if unique.issubset({0, 1}) and parsed.eq(0).mean() >= threshold:
            columns.append(column)
    return columns


# ---------------------------------------------------------------------------
# Conversión de números y fechas
# ---------------------------------------------------------------------------
def parse_number(value, decimal_mode: str = "auto") -> float:
    """Convierte '$ 1.234,56', '45%', '(120)' o '1,234.56' a float."""
    if is_missing(value) or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    text = text.replace("\u00a0", "").replace(" ", "")
    had_currency = any(sym in text for sym in _CURRENCY_SYMBOLS)
    for sym in _CURRENCY_SYMBOLS:
        text = text.replace(sym, "")

    lowered = text.lower()
    for word in _CURRENCY_WORDS:
        lowered = lowered.replace(word, "")
    text = lowered

    if text.endswith("%"):
        text = text[:-1]
    if not text or not re.fullmatch(r"[+-]?[\d.,]+", text):
        return np.nan

    has_dot = "." in text
    has_comma = "," in text

    if decimal_mode == "comma":
        text = text.replace(".", "").replace(",", ".")
    elif decimal_mode == "dot":
        text = text.replace(",", "")
    else:  # auto
        if has_dot and has_comma:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif has_comma:
            if text.count(",") > 1:
                text = text.replace(",", "")
            elif had_currency and re.fullmatch(r"[+-]?\d{1,3},\d{3}", text):
                text = text.replace(",", "")
            else:
                text = text.replace(",", ".")
        elif has_dot:
            if text.count(".") > 1 or (
                had_currency and re.fullmatch(r"[+-]?\d{1,3}\.\d{3}", text)
            ):
                text = text.replace(".", "")

    try:
        number = float(text)
    except ValueError:
        return np.nan
    if negative:
        number = -abs(number)
    if not np.isfinite(number):
        return np.nan
    return number


def numeric_ratio(series: pd.Series, decimal_mode: str = "auto", sample: int = 500) -> float:
    values = series.dropna()
    if values.empty:
        return 0.0
    unique = pd.unique(values)
    if len(unique) > sample:
        unique = unique[:sample]
    parsed = [parse_number(v, decimal_mode) for v in unique]
    ok = sum(1 for p in parsed if not pd.isna(p))
    return ok / len(unique)


def parse_date(value, dayfirst: bool = True):
    if is_missing(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        try:
            return pd.Timestamp(value)
        except (ValueError, TypeError):
            return pd.NaT
    if isinstance(value, (int, float, np.integer, np.floating)):
        # Fechas serializadas de Excel.
        if 1 <= float(value) <= 100000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value))
        return pd.NaT

    text = str(value).strip()
    if not text:
        return pd.NaT

    spanish = re.fullmatch(
        r"(\d{1,2})\s+de\s+([a-záéíóúñ]+)\s+de\s+(\d{4})", text, re.IGNORECASE
    )
    if spanish:
        month = _ES_MONTHS.get(spanish.group(2).lower())
        if month:
            text = f"{spanish.group(1)}/{month}/{spanish.group(3)}"

    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    elif re.fullmatch(r"\d{4}[-/]\d{1,2}", text):
        text = f"{text}-01"

    if not _DATE_LIKE_RE.search(text):
        return pd.NaT
    iso_like = bool(re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", text))
    try:
        return pd.Timestamp(
            date_parser.parse(text, dayfirst=False if iso_like else dayfirst, fuzzy=False)
        )
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def date_ratio(series: pd.Series, dayfirst: bool = True, sample: int = 200) -> float:
    values = series.dropna()
    if values.empty:
        return 0.0
    unique = pd.unique(values)
    if len(unique) > sample:
        unique = unique[:sample]
    parsed = [parse_date(v, dayfirst) for v in unique]
    ok = sum(1 for p in parsed if not pd.isna(p))
    return ok / len(unique)


def _has_date_hint(column: str) -> bool:
    tokens = set(str(column).lower().split("_"))
    return any(hint in tokens for hint in DATE_HINTS)


def _looks_like_id(column: str, series: pd.Series) -> bool:
    tokens = set(str(column).lower().split("_"))
    if tokens.intersection(ID_HINTS):
        return True
    values = series.dropna().astype(str)
    if values.empty:
        return False
    if values.str.fullmatch(r"0\d+").mean() > 0.7:
        return True
    if values.str.fullmatch(r"\d{6,}").mean() > 0.7:
        return True
    return False


def _convert_numeric(series: pd.Series, decimal_mode: str) -> tuple[pd.Series, int]:
    unique = pd.unique(series.dropna())
    mapping = {v: parse_number(v, decimal_mode) for v in unique}
    converted = series.map(lambda v: np.nan if is_missing(v) else mapping.get(v, np.nan))
    failed = int(converted.isna().sum() - series.isna().sum())
    return pd.to_numeric(converted, errors="coerce"), max(failed, 0)


def _convert_dates(series: pd.Series, dayfirst: bool) -> tuple[pd.Series, int]:
    unique = pd.unique(series.dropna())
    mapping = {v: parse_date(v, dayfirst) for v in unique}
    converted = series.map(lambda v: pd.NaT if is_missing(v) else mapping.get(v, pd.NaT))
    parsed = pd.to_datetime(converted, errors="coerce")
    failed = int(parsed.isna().sum() - series.isna().sum())
    return parsed, max(failed, 0)


# ---------------------------------------------------------------------------
# Pasos de limpieza
# ---------------------------------------------------------------------------
def _clean_text_columns(df: pd.DataFrame, cfg: CleaningConfig, log: list[dict]) -> pd.DataFrame:
    cols = text_columns(df)
    if not cols:
        return df
    changed = 0
    for col in cols:
        before = df[col].copy()

        def _clean(value):
            if is_missing(value):
                return pd.NA
            if not isinstance(value, str):
                text = str(value)
            else:
                text = value
            text = text.replace("\u00a0", " ")
            text = re.sub(r"\s+", " ", text).strip()
            if cfg.remove_accents:
                text = remove_accents(text)
            if cfg.case_mode == "lower":
                text = text.lower()
            elif cfg.case_mode == "upper":
                text = text.upper()
            elif cfg.case_mode == "title":
                text = text.title()
            if cfg.replace_placeholders and text.strip().lower() in PLACEHOLDER_NULLS:
                return pd.NA
            return text

        df[col] = df[col].map(_clean)
        if not df[col].equals(before):
            changed += 1

    if changed:
        modes = []
        if cfg.replace_placeholders:
            modes.append("nulos encubiertos")
        if cfg.case_mode != "none":
            modes.append({"lower": "minúsculas", "upper": "mayúsculas", "title": "título"}[cfg.case_mode])
        if cfg.remove_accents:
            modes.append("sin acentos")
        if not modes:
            modes.append("espacios")
        _log(
            log,
            "Texto",
            f"Se normalizaron espacios/valores en {changed} columnas ({', '.join(modes)}).",
            cols=changed,
        )
    return df


def _convert_types(df: pd.DataFrame, cfg: CleaningConfig, log: list[dict]) -> pd.DataFrame:
    if not cfg.auto_types and not (cfg.force_numeric_cols or cfg.force_date_cols):
        return df

    force_numeric = [c for c in cfg.force_numeric_cols if c in df.columns]
    force_date = [c for c in cfg.force_date_cols if c in df.columns]
    skip_numeric = set(cfg.skip_numeric_cols)
    skip_date = set(cfg.skip_date_cols)

    numeric_converted: list[str] = []
    date_converted: list[str] = []
    skipped_ids: list[str] = []

    for col in list(df.columns):
        series = df[col]
        if is_numeric_col(series) or is_date_col(series):
            continue
        if not is_text_col(series):
            continue
        if col in force_date:
            converted, failed = _convert_dates(series, cfg.dayfirst)
            ratio = 1 - (converted.isna().sum() / max(series.notna().sum(), 1))
            if ratio >= 0.5:
                df[col] = converted
                date_converted.append(col)
                if failed:
                    _log(
                        log,
                        "Tipos",
                        f"'{col}' convertida a fecha; {failed} valores no reconocidos quedaron nulos.",
                        cols=1,
                    )
            else:
                _log(
                    log,
                    "Tipos",
                    f"No se pudo convertir '{col}' a fecha (solo {ratio:.0%} válidos).",
                    cols=1,
                )
            continue
        if col in force_numeric:
            converted, failed = _convert_numeric(series, cfg.decimal_mode)
            ratio = 1 - (converted.isna().sum() / max(series.notna().sum(), 1))
            if ratio >= 0.5:
                df[col] = converted
                numeric_converted.append(col)
                if failed:
                    _log(
                        log,
                        "Tipos",
                        f"'{col}' convertida a número; {failed} valores no numéricos quedaron nulos.",
                        cols=1,
                    )
            else:
                _log(
                    log,
                    "Tipos",
                    f"No se pudo convertir '{col}' a número (solo {ratio:.0%} válidos).",
                    cols=1,
                )
            continue
        if not cfg.auto_types:
            continue
        if col in skip_numeric or col in skip_date:
            continue
        if _looks_like_id(col, series):
            skipped_ids.append(col)
            continue

        num_ratio = numeric_ratio(series, cfg.decimal_mode)
        is_date_candidate = _has_date_hint(col) or _DATE_LIKE_RE.search(
            " ".join(series.dropna().astype(str).head(30).tolist())
        )
        dat_ratio = date_ratio(series, cfg.dayfirst) if is_date_candidate else 0.0

        if dat_ratio >= 0.8 and (dat_ratio >= num_ratio or _has_date_hint(col)):
            converted, failed = _convert_dates(series, cfg.dayfirst)
            df[col] = converted
            date_converted.append(col)
            if failed:
                _log(
                    log,
                    "Tipos",
                    f"'{col}' convertida a fecha; {failed} valores no reconocidos quedaron nulos.",
                    cols=1,
                )
        elif num_ratio >= 0.85:
            converted, failed = _convert_numeric(series, cfg.decimal_mode)
            df[col] = converted
            numeric_converted.append(col)
            if failed:
                _log(
                    log,
                    "Tipos",
                    f"'{col}' convertida a número; {failed} valores no numéricos quedaron nulos.",
                    cols=1,
                )

    if numeric_converted:
        _log(
            log,
            "Tipos",
            f"Texto a número: {', '.join(numeric_converted[:8])}"
            + ("…" if len(numeric_converted) > 8 else ""),
            cols=len(numeric_converted),
        )
    if date_converted:
        _log(
            log,
            "Tipos",
            f"Texto a fecha: {', '.join(date_converted[:8])}"
            + ("…" if len(date_converted) > 8 else ""),
            cols=len(date_converted),
        )
    if skipped_ids:
        _log(
            log,
            "Tipos",
            f"Se conservaron como texto posibles identificadores: {', '.join(skipped_ids[:6])}"
            + ("…" if len(skipped_ids) > 6 else ""),
            cols=len(skipped_ids),
        )
    return df


def _impute(df: pd.DataFrame, cfg: CleaningConfig, log: list[dict]) -> pd.DataFrame:
    exclude = set(cfg.impute_exclude)
    num_cols = [c for c in numeric_columns(df) if c not in exclude and df[c].isna().any()]
    cat_cols = [c for c in text_columns(df) if c not in exclude and df[c].isna().any()]
    dat_cols = [c for c in date_columns(df) if c not in exclude and df[c].isna().any()]

    drop_rows: list[str] = []
    drop_cols: list[str] = []

    if cfg.numeric_strategy == "drop_rows":
        drop_rows.extend(num_cols)
    elif cfg.numeric_strategy == "drop_cols":
        drop_cols.extend(num_cols)
    if cfg.categorical_strategy == "drop_rows":
        drop_rows.extend(cat_cols)
    elif cfg.categorical_strategy == "drop_cols":
        drop_cols.extend(cat_cols)

    if drop_cols:
        df = df.drop(columns=list(dict.fromkeys(drop_cols)))
        _log(
            log,
            "Nulos",
            f"Se eliminaron {len(set(drop_cols))} columnas con valores nulos.",
            cols=len(set(drop_cols)),
        )
        num_cols = [c for c in num_cols if c in df.columns]
        cat_cols = [c for c in cat_cols if c in df.columns]
        dat_cols = [c for c in dat_cols if c in df.columns]

    if drop_rows:
        before = len(df)
        df = df.dropna(subset=list(dict.fromkeys(drop_rows)))
        removed = before - len(df)
        if removed:
            _log(
                log,
                "Nulos",
                f"Se eliminaron {removed} filas con nulos en columnas clave.",
                rows=removed,
                cols=len(set(drop_rows)),
            )

    filled_num = 0
    filled_cat = 0
    filled_date = 0

    if cfg.numeric_strategy in {"median", "mean", "zero", "ffill"}:
        for col in num_cols:
            s = df[col]
            if not s.isna().any():
                continue
            if cfg.numeric_strategy == "median":
                value = s.median()
            elif cfg.numeric_strategy == "mean":
                value = s.mean()
            elif cfg.numeric_strategy == "zero":
                value = 0
            else:
                value = None
            if cfg.numeric_strategy == "ffill":
                df[col] = s.ffill()
            else:
                if pd.isna(value):
                    continue
                df[col] = s.fillna(value)
            filled_num += 1

    if cfg.categorical_strategy in {"mode", "constant", "ffill"}:
        for col in cat_cols:
            s = df[col]
            if not s.isna().any():
                continue
            if cfg.categorical_strategy == "mode":
                mode = s.mode(dropna=True)
                if mode.empty:
                    continue
                df[col] = s.fillna(mode.iloc[0])
            elif cfg.categorical_strategy == "constant":
                df[col] = s.fillna(cfg.constant_value)
            else:
                df[col] = s.ffill()
            filled_cat += 1

    for col in dat_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill()
            filled_date += 1

    if filled_num:
        labels = {
            "median": "mediana",
            "mean": "media",
            "zero": "cero",
            "ffill": "valor anterior",
        }
        _log(
            log,
            "Nulos",
            f"Nulos numéricos imputados con {labels.get(cfg.numeric_strategy, cfg.numeric_strategy)} en {filled_num} columnas.",
            cols=filled_num,
        )
    if filled_cat:
        labels = {"mode": "moda", "constant": cfg.constant_value, "ffill": "valor anterior"}
        _log(
            log,
            "Nulos",
            f"Nulos categóricos imputados con {labels.get(cfg.categorical_strategy, cfg.categorical_strategy)} en {filled_cat} columnas.",
            cols=filled_cat,
        )
    if filled_date:
        _log(
            log,
            "Nulos",
            f"Fechas nulas completadas con el valor anterior en {filled_date} columnas.",
            cols=filled_date,
        )
    return df


def _handle_outliers(df: pd.DataFrame, cfg: CleaningConfig, log: list[dict]) -> pd.DataFrame:
    if cfg.outlier_method == "none" or cfg.outlier_action == "none":
        return df
    candidates = cfg.outlier_cols or numeric_columns(df)
    cols = [c for c in candidates if c in df.columns and is_numeric_col(df[c])]
    if not cols:
        return df

    total_cells = 0
    total_removed = 0
    details: list[str] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 4:
            continue
        if cfg.outlier_method == "iqr":
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            low = q1 - cfg.outlier_factor * iqr
            high = q3 + cfg.outlier_factor * iqr
        else:  # zscore
            mean, std = valid.mean(), valid.std()
            if not std or pd.isna(std) or std == 0:
                continue
            low = mean - cfg.outlier_factor * std
            high = mean + cfg.outlier_factor * std
        mask = ((s < low) | (s > high)).fillna(False)
        count = int(mask.sum())
        if count == 0:
            continue
        total_cells += count
        details.append(f"{col} ({count})")
        if cfg.outlier_action == "clip":
            df[col] = s.clip(lower=low, upper=high)
        elif cfg.outlier_action == "remove":
            df = df.loc[~mask]
            total_removed += count
        else:  # flag
            flag_col = f"{col}_atipico"
            df[flag_col] = mask.astype(int)

    if total_cells:
        action = {
            "clip": "recortados",
            "remove": "eliminados",
            "flag": "marcados",
        }[cfg.outlier_action]
        shown = ", ".join(details[:6]) + ("…" if len(details) > 6 else "")
        _log(
            log,
            "Atípicos",
            f"{total_cells} valores {action} ({cfg.outlier_method.upper()}, factor {cfg.outlier_factor:g}): {shown}",
            rows=total_removed,
            cols=len(details),
        )
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def apply_cleaning(source: pd.DataFrame, cfg: CleaningConfig) -> CleaningResult:
    """Aplica el pipeline de limpieza y devuelve el dataset y su bitácora."""
    if source is None or source.empty:
        raise ValueError("No hay datos para limpiar.")

    df = source.copy()
    log: list[dict] = []
    rows_before, cols_before = df.shape
    missing_before = float(df.isna().mean().mean()) if df.size else 0.0
    column_map: dict[str, str] = {}

    _log(
        log,
        "Inicio",
        f"Dataset original: {rows_before:,} filas × {cols_before:,} columnas.",
    )

    # 1. Nombres de columnas
    if cfg.normalize_columns:
        new_names, column_map = unique_column_names([str(c) for c in df.columns])
        renamed = sum(1 for old, new in column_map.items() if old != new)
        df.columns = new_names
        if renamed:
            examples = [
                f"{old} → {new}"
                for old, new in list(column_map.items())[:4]
                if old != new
            ]
            _log(
                log,
                "Estructura",
                f"{renamed} nombres normalizados. Ej.: " + "; ".join(examples),
                cols=renamed,
            )

    if cfg.drop_sensitive_cols:
        selected = (
            detect_sensitive_columns(df)
            if cfg.sensitive_cols is None
            else [column_map.get(str(column), str(column)) for column in cfg.sensitive_cols]
        )
        to_drop = [column for column in selected if column in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            shown = ", ".join(to_drop[:8]) + ("…" if len(to_drop) > 8 else "")
            _log(
                log,
                "Privacidad",
                f"Se eliminaron {len(to_drop)} columnas identificadas como credenciales: {shown}.",
                cols=len(to_drop),
            )

    # 2. Filas/columnas totalmente vacías
    if cfg.drop_empty_rows:
        before = len(df)
        df = df.dropna(axis=0, how="all")
        if len(df) != before:
            _log(
                log,
                "Estructura",
                f"Se eliminaron {before - len(df)} filas totalmente vacías.",
                rows=before - len(df),
            )
    if cfg.drop_empty_cols:
        before = len(df.columns)
        df = df.dropna(axis=1, how="all")
        removed = before - len(df.columns)
        if removed:
            _log(log, "Estructura", f"Se eliminaron {removed} columnas vacías.", cols=removed)

    # 3. Texto y nulos encubiertos
    if cfg.clean_text or cfg.replace_placeholders:
        df = _clean_text_columns(df, cfg, log)

    # 4. Tipos
    df = _convert_types(df, cfg, log)

    # 5. Columnas con demasiados nulos
    if cfg.drop_high_null_cols and cfg.high_null_threshold < 1:
        null_share = df.isna().mean()
        to_drop = [c for c in df.columns if null_share[c] >= cfg.high_null_threshold]
        if to_drop:
            df = df.drop(columns=to_drop)
            _log(
                log,
                "Estructura",
                f"Se eliminaron {len(to_drop)} columnas con ≥ {cfg.high_null_threshold:.0%} de nulos.",
                cols=len(to_drop),
            )

    # 6. Duplicados
    if cfg.drop_duplicates:
        subset = cfg.duplicate_subset or None
        if subset:
            subset = [c for c in subset if c in df.columns]
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        removed = before - len(df)
        if removed:
            criterion = "columnas seleccionadas" if subset else "todas las columnas"
            _log(
                log,
                "Duplicados",
                f"Se eliminaron {removed} filas duplicadas ({criterion}).",
                rows=removed,
            )

    # 7. Nulos
    df = _impute(df, cfg, log)

    # 8. Columnas sin información numérica distinta de cero o binarias dispersas
    if cfg.drop_zero_cols:
        zero_only = _zero_only_columns(df)
        zero_dominant = _zero_dominant_binary_columns(
            df, cfg.zero_dominance_threshold
        )
        to_drop = list(dict.fromkeys(zero_only + zero_dominant))
        if to_drop:
            df = df.drop(columns=to_drop)
            shown = ", ".join(to_drop[:8]) + ("…" if len(to_drop) > 8 else "")
            _log(
                log,
                "Estructura",
                f"Se eliminaron {len(to_drop)} columnas con todos cero o al menos "
                f"{cfg.zero_dominance_threshold:.0%} de ceros en variables binarias: {shown}.",
                cols=len(to_drop),
            )

    # 9. Atípicos
    df = _handle_outliers(df, cfg, log)

    # 10. Columnas constantes
    if cfg.drop_constant_cols:
        constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant:
            df = df.drop(columns=constant)
            _log(
                log,
                "Estructura",
                f"Se eliminaron {len(constant)} columnas constantes: {', '.join(constant[:6])}"
                + ("…" if len(constant) > 6 else ""),
                cols=len(constant),
            )

    df = df.reset_index(drop=True)
    rows_after, cols_after = df.shape
    missing_after = float(df.isna().mean().mean()) if df.size else 0.0

    if not log:
        _log(log, "Fin", "No se aplicaron transformaciones.")
    _log(
        log,
        "Fin",
        f"Resultado: {rows_after:,} filas × {cols_after:,} columnas "
        f"({rows_before - rows_after:,} filas menos).",
    )

    stats = {
        "filas_antes": rows_before,
        "columnas_antes": cols_before,
        "filas_despues": rows_after,
        "columnas_despues": cols_after,
        "filas_eliminadas": rows_before - rows_after,
        "columnas_eliminadas": cols_before - cols_after,
        "nulos_antes": missing_before,
        "nulos_despues": missing_after,
        "duplicados_eliminados": int(
            next((entry["Filas afectadas"] for entry in log if entry["Paso"] == "Duplicados"), 0)
        ),
    }

    log_df = pd.DataFrame(log, columns=LOG_COLUMNS)
    log_df.insert(0, "#", range(1, len(log_df) + 1))
    return CleaningResult(
        df=df,
        log=log_df,
        stats=stats,
        column_map=column_map,
        config=asdict(cfg),
    )
