"""Prueba de humo del motor de limpieza, gráficas y exportaciones.

Se ejecuta con:  .venv\\Scripts\\python.exe tests\\smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.charts import (  # noqa: E402
    box_by_category,
    category_bar,
    correlation_heatmap,
    custom_chart,
    donut_chart,
    empty_figure,
    missing_values_chart,
    numeric_distribution,
    numeric_grid,
    pareto_chart,
    scatter_plot,
    time_series,
    treemap_chart,
)
from src.cleaning import (  # noqa: E402
    CleaningConfig,
    apply_cleaning,
    detect_sensitive_columns,
    parse_date,
    parse_number,
)
from src.insights import executive_summary, generate_insights  # noqa: E402
from src.profiling import (  # noqa: E402
    column_roles,
    dataset_summary,
    detect_business_metrics,
    diagnose,
    profile_columns,
    quality_components,
)
from src.report import build_csv, build_excel, build_html_report  # noqa: E402
from src.sample_data import build_sample_dataset, save_sample_csv  # noqa: E402


def test_sensitive_and_zero_columns() -> None:
    secrets = [f"secret-{index}" for index in range(10)]
    source = pd.DataFrame(
        {
            "All Zero": ["0", "0,0", "0.00", "0", "0", "0", "0", "0", "0", "0"],
            "Sparse Flag": [0] * 9 + [1],
            "Balanced Flag": [0] * 5 + [1] * 5,
            "Mixed Zero": [0] * 5 + [2] * 5,
            "Password": secrets,
            "tokenrecuperar": [f"token-{index}" for index in range(10)],
            "Valor": list(range(10)),
        }
    )
    assert set(detect_sensitive_columns(source)) == {"Password", "tokenrecuperar"}

    config = CleaningConfig(
        drop_duplicates=False,
        drop_constant_cols=False,
        numeric_strategy="none",
        categorical_strategy="none",
        outlier_method="none",
    )
    result = apply_cleaning(source, config)
    assert "all_zero" not in result.df.columns
    assert "sparse_flag" not in result.df.columns
    assert "balanced_flag" in result.df.columns
    assert "mixed_zero" in result.df.columns
    assert "password" not in result.df.columns
    assert "tokenrecuperar" not in result.df.columns
    assert not result.df.astype(str).isin(secrets).to_numpy().any()
    assert not result.log.astype(str).isin(secrets).to_numpy().any()

    conservative = apply_cleaning(
        source,
        CleaningConfig(
            drop_duplicates=False,
            drop_constant_cols=False,
            zero_dominance_threshold=0.95,
            numeric_strategy="none",
            categorical_strategy="none",
            outlier_method="none",
        ),
    )
    assert "sparse_flag" in conservative.df.columns

    preserved = apply_cleaning(
        source,
        CleaningConfig(
            drop_duplicates=False,
            drop_constant_cols=False,
            drop_zero_cols=False,
            drop_sensitive_cols=False,
            numeric_strategy="none",
            categorical_strategy="none",
            outlier_method="none",
        ),
    )
    assert "all_zero" in preserved.df.columns
    assert "password" in preserved.df.columns
    print("[OK] Eliminación de columnas con solo ceros y credenciales")


def main() -> None:
    test_sensitive_and_zero_columns()

    raw = build_sample_dataset()
    assert raw.shape[0] > 1000, "El dataset de ejemplo es demasiado pequeño"
    assert "Fecha" in raw.columns

    # --- Conversión de valores -------------------------------------------------
    assert abs(parse_number("$1.234,56") - 1234.56) < 0.001
    assert abs(parse_number("1,234.56") - 1234.56) < 0.001
    assert abs(parse_number("45%") - 45) < 0.001
    assert pd.isna(parse_number("3 unidades"))
    assert pd.isna(parse_number("N/D"))
    assert parse_date("05/01/2024") == pd.Timestamp("2024-01-05")
    assert parse_date("2024-01-05") == pd.Timestamp("2024-01-05")
    assert parse_date("5 de enero de 2024") == pd.Timestamp("2024-01-05")
    print("[OK] Conversión de números y fechas")

    # --- Limpieza --------------------------------------------------------------
    cfg = CleaningConfig(outlier_action="clip", drop_high_null_cols=False)
    result = apply_cleaning(raw, cfg)
    clean = result.df
    assert clean.shape[0] <= raw.shape[0]
    assert str(clean["fecha"].dtype).startswith("datetime"), clean["fecha"].dtype
    assert pd.api.types.is_numeric_dtype(clean["ingreso"]), clean["ingreso"].dtype
    assert pd.api.types.is_numeric_dtype(clean["precio_unitario"]), clean["precio_unitario"].dtype
    assert "pais" not in clean.columns, "La columna constante debe eliminarse"
    assert clean.duplicated().sum() == 0
    assert not result.log.empty
    print(f"[OK] Limpieza: {raw.shape} → {clean.shape}, bitácora con {len(result.log)} pasos")

    # --- Perfilado -------------------------------------------------------------
    summary = dataset_summary(clean)
    quality = quality_components(clean)
    raw_quality = quality_components(raw)
    roles = detect_business_metrics(clean)
    all_roles = column_roles(clean)
    assert 0 <= quality["puntaje"] <= 100
    assert raw_quality["puntaje"] < quality["puntaje"] - 5, (
        f"La calidad de los datos crudos no debería ser casi perfecta: "
        f"{raw_quality['puntaje']:.1f} vs {quality['puntaje']:.1f}"
    )
    assert raw_quality["consistencia"] < 0.8, raw_quality
    assert summary["filas"] == len(clean)
    assert roles["metric"] is not None
    assert roles["date"] == "fecha"
    assert not diagnose(raw).empty
    assert not profile_columns(clean).empty
    print(
        f"[OK] Perfilado: calidad {raw_quality['puntaje']:.1f} → {quality['puntaje']:.1f}/100, "
        f"métrica sugerida '{roles['metric']}'"
    )

    # --- Gráficas --------------------------------------------------------------
    metric = roles["metric"]
    dim = roles["dimension"]
    figures = [
        missing_values_chart(raw),
        numeric_distribution(clean, metric),
        numeric_grid(clean, all_roles["numeric"][:4]),
        correlation_heatmap(clean),
        category_bar(clean, dim, metric),
        pareto_chart(clean, dim, metric),
        donut_chart(clean, dim),
        treemap_chart(clean, [dim, "producto"], metric),
        scatter_plot(clean, "unidades", metric, color=dim),
        box_by_category(clean, dim, metric),
        time_series(clean, "fecha", metric, freq="MS"),
        custom_chart(clean, "Barras", metric, dim),
        empty_figure(),
    ]
    for fig in figures:
        assert fig is not None
        assert len(fig.to_dict().get("data", [])) >= 0
    print(f"[OK] {len(figures)} gráficas generadas sin errores")

    # --- Hallazgos y exportaciones ---------------------------------------------
    insights = generate_insights(clean, result.stats)
    assert insights
    summary_text = executive_summary(clean, result.stats)
    assert "registros" in summary_text

    csv_bytes = build_csv(clean)
    excel_bytes = build_excel(clean, result.log, result.stats, dataset_name="ventas_demo")
    html_report = build_html_report(
        df=clean,
        dataset_name="ventas_demo.csv",
        stats=result.stats,
        log=result.log,
        insights=insights,
        figures=[("Calidad de datos", figures[0]), ("Evolución", figures[10])],
        summary_text=summary_text,
    )
    assert len(csv_bytes) > 1000
    assert excel_bytes[:2] == b"PK"
    assert "<html" in html_report and "Hallazgos" in html_report
    print(f"[OK] Exportaciones: CSV {len(csv_bytes)/1024:.0f} KB, Excel {len(excel_bytes)/1024:.0f} KB, HTML {len(html_report)/1024:.0f} KB")

    sample_path = save_sample_csv(ROOT / "sample_data" / "ventas_demo.csv")
    print(f"[OK] Dataset de ejemplo: {sample_path}")

    print("\nTodo correcto. La app está lista para ejecutarse con: streamlit run app.py")


if __name__ == "__main__":
    main()
