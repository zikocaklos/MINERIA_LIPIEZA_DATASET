"""Procesamiento por lotes: limpia un dataset y genera los mismos entregables.

Uso:
    python procesar.py "datos.xlsx" --salida "resultados"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.charts import (
    category_bar,
    correlation_heatmap,
    missing_values_chart,
    numeric_grid,
    pareto_chart,
    time_series,
)
from src.cleaning import CleaningConfig, apply_cleaning
from src.insights import executive_summary, generate_insights
from src.io_utils import load_dataset
from src.profiling import dataset_summary, detect_business_metrics, quality_components
from src.report import build_csv, build_excel, build_html_report
from src.utils import numeric_columns, safe_filename


def build_figures(df: pd.DataFrame) -> list[tuple[str, object]]:
    roles = detect_business_metrics(df)
    nums = numeric_columns(df)
    metric, dim, date_col = roles.get("metric"), roles.get("dimension"), roles.get("date")
    figures: list[tuple[str, object]] = [("Calidad: valores nulos", missing_values_chart(df))]
    if nums:
        figures.append(("Distribución de variables", numeric_grid(df, nums[:6])))
    if len(nums) >= 2:
        figures.append(("Correlación entre variables", correlation_heatmap(df)))
    if dim and metric:
        figures.append(("Categorías con mayor aporte", category_bar(df, dim, metric)))
        figures.append(("Pareto por categoría", pareto_chart(df, dim, metric)))
    if date_col and metric:
        figures.append(("Evolución en el tiempo", time_series(df, date_col, metric, freq="MS")))
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Limpia un dataset y genera CSV, Excel e informe HTML."
    )
    parser.add_argument("entrada", help="Archivo CSV, Excel, JSON o Parquet")
    parser.add_argument("-o", "--salida", default="salida", help="Carpeta de resultados")
    parser.add_argument(
        "--decimal",
        choices=["auto", "comma", "dot"],
        default="auto",
        help="Interpretación de separadores numéricos",
    )
    parser.add_argument(
        "--sin-atipicos",
        action="store_true",
        help="No recortar valores atípicos",
    )
    args = parser.parse_args()

    source = Path(args.entrada)
    if not source.exists():
        print(f"No existe el archivo: {source}")
        return 1

    print(f"Leyendo {source.name}...")
    loaded = load_dataset(source.read_bytes(), source.name, excel_sheet=0)
    raw = loaded.df
    print(f"  {len(raw):,} filas × {raw.shape[1]} columnas".replace(",", "."))

    cfg = CleaningConfig(
        decimal_mode=args.decimal,
        outlier_method="none" if args.sin_atipicos else "iqr",
        outlier_action="clip",
    )
    print("Aplicando limpieza...")
    result = apply_cleaning(raw, cfg)
    clean = result.df
    quality = quality_components(clean)
    print(
        f"  {result.stats['filas_antes']:,} → {result.stats['filas_despues']:,} filas · "
        f"calidad {quality['puntaje']:.0f}/100".replace(",", ".")
    )

    out_dir = Path(args.salida)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem

    csv_path = out_dir / safe_filename(f"{stem}_limpio", ".csv")
    csv_path.write_bytes(build_csv(clean))
    print(f"  CSV:   {csv_path}")

    excel_path = out_dir / safe_filename(f"{stem}_informe", ".xlsx")
    excel_path.write_bytes(build_excel(clean, result.log, result.stats, dataset_name=source.name))
    print(f"  Excel: {excel_path}")

    insights = generate_insights(clean, result.stats)
    html_path = out_dir / safe_filename(f"{stem}_informe", ".html")
    html_path.write_text(
        build_html_report(
            df=clean,
            dataset_name=source.name,
            stats=result.stats,
            log=result.log,
            insights=insights,
            figures=build_figures(clean),
            summary_text=executive_summary(clean, result.stats),
        ),
        encoding="utf-8",
    )
    print(f"  HTML:  {html_path}")

    print("\nHallazgos principales:")
    for insight in insights[:5]:
        print(f"  - {insight['titulo']}")
    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
