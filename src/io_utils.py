"""Carga de archivos CSV, Excel, JSON y Parquet."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PLACEHOLDER_NULLS

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_SEPARATORS = (",", ";", "\t", "|")


@dataclass
class LoadResult:
    df: pd.DataFrame
    sheet: str | None = None
    encoding: str | None = None
    separator: str | None = None


def detect_encoding(data: bytes) -> str:
    """Devuelve la primera codificación que decodifica el archivo sin errores."""
    for enc in _ENCODINGS:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_separator(data: bytes, encoding: str) -> str:
    """Detecta el separador más probable de un archivo delimitado."""
    try:
        text = data.decode(encoding, errors="replace")[:20000]
    except Exception:
        text = data.decode("latin-1", errors="replace")[:20000]
    sample = "\n".join(line for line in text.splitlines()[:20] if line.strip())
    if sample:
        try:
            return csv.Sniffer().sniff(sample, delimiters="".join(_SEPARATORS)).delimiter
        except csv.Error:
            pass
    first = sample.splitlines()[0] if sample else ""
    counts = {sep: first.count(sep) for sep in _SEPARATORS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def list_excel_sheets(data: bytes) -> list[str]:
    """Lista las hojas de un libro de Excel sin cargar todos los datos."""
    try:
        with pd.ExcelFile(io.BytesIO(data), engine="openpyxl") as xls:
            return [str(s) for s in xls.sheet_names]
    except Exception:
        try:
            with pd.ExcelFile(io.BytesIO(data)) as xls:
                return [str(s) for s in xls.sheet_names]
        except Exception:
            return []


def read_csv_bytes(data: bytes, encoding: str | None = None, separator: str | None = None) -> LoadResult:
    enc = encoding if encoding and encoding != "auto" else detect_encoding(data)
    sep = separator if separator and separator != "auto" else detect_separator(data, enc)
    last_error: Exception | None = None
    for candidate in dict.fromkeys([sep, ";", ",", "\t", "|"]):
        try:
            df = pd.read_csv(
                io.BytesIO(data),
                sep=candidate,
                encoding=enc,
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
            if df.shape[1] > 1 or candidate == sep:
                if df.shape[1] > 1 or len(df.columns) >= 1:
                    return LoadResult(df=df, encoding=enc, separator=candidate)
        except Exception as exc:  # pragma: no cover - depende del archivo
            last_error = exc
    if last_error:
        raise ValueError(f"No se pudo leer el CSV: {last_error}")
    raise ValueError("No se pudo leer el CSV.")


def read_excel_bytes(data: bytes, sheet: str | int | None = 0, header_row: int = 0) -> LoadResult:
    try:
        df = pd.read_excel(
            io.BytesIO(data),
            sheet_name=sheet if sheet is not None else 0,
            header=header_row,
        )
    except Exception as exc:
        raise ValueError(f"No se pudo leer el archivo de Excel: {exc}") from exc
    sheet_name = str(sheet) if isinstance(sheet, str) else None
    return LoadResult(df=df, sheet=sheet_name)


def read_json_bytes(data: bytes) -> LoadResult:
    try:
        df = pd.read_json(io.BytesIO(data))
    except Exception:
        try:
            df = pd.read_json(io.BytesIO(data), lines=True)
        except Exception:
            try:
                payload = json.loads(data.decode("utf-8-sig", errors="replace"))
                df = pd.json_normalize(payload)
            except Exception as exc:
                raise ValueError(f"No se pudo leer el JSON: {exc}") from exc
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El JSON no contiene una tabla válida.")
    return LoadResult(df=df)


def read_parquet_bytes(data: bytes) -> LoadResult:
    try:
        df = pd.read_parquet(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"No se pudo leer el Parquet: {exc}") from exc
    return LoadResult(df=df)


def load_dataset(
    data: bytes,
    filename: str,
    *,
    excel_sheet: str | int | None = 0,
    encoding: str | None = None,
    separator: str | None = None,
) -> LoadResult:
    """Carga un dataset según la extensión del archivo."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".csv", ".txt", ".tsv", ".dat"}:
        result = read_csv_bytes(data, encoding=encoding, separator=separator)
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        result = read_excel_bytes(data, sheet=excel_sheet)
    elif suffix == ".json":
        result = read_json_bytes(data)
    elif suffix in {".parquet", ".pq"}:
        result = read_parquet_bytes(data)
    else:
        raise ValueError(
            f"Formato '{suffix or 'desconocido'}' no soportado. "
            "Usa CSV, Excel (.xlsx/.xls), JSON o Parquet."
        )

    df = result.df
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("El archivo no contiene una tabla válida.")
    # Elimina columnas y filas totalmente vacías que suelen aparecer al final
    # de los archivos exportados desde Excel.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[0] == 0:
        raise ValueError("El dataset no contiene filas de datos.")
    return LoadResult(
        df=df.reset_index(drop=True),
        sheet=result.sheet,
        encoding=result.encoding,
        separator=result.separator,
    )


def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte textos que representan nulos ('N/A', '-', ...) en valores nulos."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        s = out[col]
        mask = s.map(
            lambda v: isinstance(v, str) and v.strip().lower() in PLACEHOLDER_NULLS
        )
        if mask.any():
            out.loc[mask, col] = pd.NA
    return out
