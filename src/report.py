"""Exportación: CSV, Excel con formato e informe HTML autocontenido."""

from __future__ import annotations

import html
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.io as pio

from .profiling import dataset_summary, profile_columns, quality_components
from .utils import fmt_number, fmt_pct


# ---------------------------------------------------------------------------
# CSV / Excel
# ---------------------------------------------------------------------------
def build_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]) and getattr(out[col].dt, "tz", None) is not None:
            out[col] = out[col].dt.tz_localize(None)
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: v if not isinstance(v, (list, dict, set)) else str(v)
            )
    return out


def build_excel(
    df: pd.DataFrame,
    log: pd.DataFrame,
    stats: dict,
    *,
    dataset_name: str = "dataset",
) -> bytes:
    buffer = io.BytesIO()
    clean = _excel_safe(df)
    profile = profile_columns(df)
    summary = dataset_summary(df)
    quality = quality_components(df)

    summary_rows = [
        ("Dataset", dataset_name),
        ("Generado", datetime.now().strftime("%d/%m/%Y %H:%M")),
        ("Filas finales", summary["filas"]),
        ("Columnas finales", summary["columnas"]),
        ("Filas originales", stats.get("filas_antes", "")),
        ("Columnas originales", stats.get("columnas_antes", "")),
        ("Filas eliminadas", stats.get("filas_eliminadas", "")),
        ("Columnas eliminadas", stats.get("columnas_eliminadas", "")),
        ("Duplicados eliminados", stats.get("duplicados_eliminados", "")),
        ("Celdas vacías (%)", f"{summary['nulos_pct']:.1%}"),
        ("Puntaje de calidad (/100)", f"{quality['puntaje']:.0f}"),
        ("Completitud", f"{quality['completitud']:.1%}"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Indicador", "Valor"])
    log_sheet = (
        log
        if log is not None and not log.empty
        else pd.DataFrame({"Bitácora": ["Sin transformaciones registradas"]})
    )

    with pd.ExcelWriter(
        buffer,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
    ) as writer:
        clean.to_excel(writer, sheet_name="Datos limpios", index=False)
        summary_df.to_excel(writer, sheet_name="Resumen", index=False)
        profile.to_excel(writer, sheet_name="Perfil de columnas", index=False)
        log_sheet.to_excel(writer, sheet_name="Bitácora de limpieza", index=False)

        workbook = writer.book
        header = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#1D4ED8",
                "font_color": "#FFFFFF",
                "border": 0,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        )
        title_fmt = workbook.add_format({"bold": True, "font_size": 14, "font_color": "#1D4ED8"})

        for sheet_name, frame in {
            "Datos limpios": clean,
            "Resumen": summary_df,
            "Perfil de columnas": profile,
            "Bitácora de limpieza": log_sheet,
        }.items():
            if frame is None:
                continue
            ws = writer.sheets[sheet_name]
            ws.set_row(0, 26)
            n_rows = max(len(frame), 1)
            n_cols = max(len(frame.columns), 1)
            ws.autofilter(0, 0, n_rows, n_cols - 1)
            ws.freeze_panes(1, 0)
            for idx, col in enumerate(frame.columns):
                try:
                    longest = int(frame[col].astype(str).str.len().max())
                except (ValueError, TypeError):
                    longest = 12
                width = max(12, min(48, (longest if pd.notna(longest) else 12) + 2))
                ws.set_column(idx, idx, width)
                ws.write(0, idx, str(col), header)

        ws = writer.sheets["Resumen"]
        ws.write(0, 4, "DataLimpia Analytics", title_fmt)

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Informe HTML
# ---------------------------------------------------------------------------
_LEVEL_STYLES = {
    "success": ("#ECFDF5", "#059669", "✅"),
    "warning": ("#FFFBEB", "#D97706", "⚠️"),
    "danger": ("#FEF2F2", "#DC2626", "🔴"),
    "info": ("#EFF6FF", "#2563EB", "💡"),
}


def _insight_html(insight: dict) -> str:
    bg, color, icon = _LEVEL_STYLES.get(insight.get("nivel", "info"), _LEVEL_STYLES["info"])
    action = insight.get("accion", "")
    action_html = (
        f'<p class="action"><strong>Acción sugerida:</strong> {html.escape(str(action))}</p>'
        if action
        else ""
    )
    return f"""
    <div class="insight" style="background:{bg};border-left:4px solid {color};">
      <div class="insight-title" style="color:{color};">{icon} {html.escape(str(insight.get('titulo', '')))}</div>
      <p>{html.escape(str(insight.get('detalle', '')))}</p>
      {action_html}
    </div>
    """


def _kpi_html(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="kpi-hint">{html.escape(hint)}</div>' if hint else ""
    return f"""
    <div class="kpi">
      <div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value">{html.escape(value)}</div>
      {hint_html}
    </div>
    """


def build_html_report(
    *,
    df: pd.DataFrame,
    dataset_name: str,
    stats: dict,
    log: pd.DataFrame,
    insights: list[dict],
    figures: list[tuple[str, object]],
    summary_text: str = "",
) -> str:
    quality = quality_components(df)
    summary = dataset_summary(df)
    generated = datetime.now().strftime("%d/%m/%Y %H:%M")
    quality_word = (
        "Excelente" if quality["puntaje"] >= 90 else
        "Buena" if quality["puntaje"] >= 75 else
        "Regular" if quality["puntaje"] >= 55 else "Baja"
    )

    kpis = "".join(
        [
            _kpi_html("Registros", fmt_number(summary["filas"], 0)),
            _kpi_html("Columnas", fmt_number(summary["columnas"], 0)),
            _kpi_html("Calidad", f"{quality['puntaje']:.0f}/100", quality_word),
            _kpi_html("Completitud", fmt_pct(quality["completitud"])),
            _kpi_html("Duplicados eliminados", fmt_number(stats.get("duplicados_eliminados", 0), 0)),
            _kpi_html("Filas eliminadas", fmt_number(stats.get("filas_eliminadas", 0), 0)),
        ]
    )

    charts_html = []
    plotly_loaded = False
    for title, fig in figures:
        if fig is None:
            continue
        try:
            chart = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=True if not plotly_loaded else False,
                config={"displayModeBar": False, "responsive": True},
            )
            plotly_loaded = True
        except Exception:
            continue
        charts_html.append(
            f'<section class="chart"><h2>{html.escape(title)}</h2>{chart}</section>'
        )

    profile = profile_columns(df).head(40)
    for col in ["% Nulos"]:
        profile[col] = profile[col].map(fmt_pct)
    for col in ["Mínimo", "Mediana", "Media", "Máximo"]:
        profile[col] = profile[col].map(
            lambda v: fmt_number(v)
            if isinstance(v, (int, float, np.number)) and pd.notna(v)
            else "—"
        )
    profile_html = profile.to_html(index=False, classes="table", border=0, na_rep="—", escape=True)

    log_html = (
        log.drop(columns=["#"], errors="ignore")
        .to_html(index=False, classes="table", border=0, na_rep="—", escape=True)
        if log is not None and not log.empty
        else "<p>Sin transformaciones registradas.</p>"
    )

    insights_html = "".join(_insight_html(i) for i in insights)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe · {html.escape(dataset_name)}</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Inter, Arial, sans-serif; margin: 0; padding: 36px 28px; background: #F6F8FC; color: #0F172A; }}
  .wrap {{ max-width: 1120px; margin: 0 auto; }}
  header {{ background: linear-gradient(135deg, #1D4ED8, #0EA5E9); color: white; padding: 34px 36px; border-radius: 18px; margin-bottom: 22px; }}
  header h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: -0.4px; }}
  header p {{ margin: 0; opacity: .9; }}
  .badge {{ display: inline-block; margin-top: 14px; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); padding: 5px 12px; border-radius: 999px; font-size: 12px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .kpi {{ background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px; }}
  .kpi-label {{ font-size: 12px; color: #64748B; text-transform: uppercase; letter-spacing: .5px; }}
  .kpi-value {{ font-size: 24px; font-weight: 700; margin-top: 4px; color: #0F172A; }}
  .kpi-hint {{ font-size: 11px; color: #94A3B8; margin-top: 2px; }}
  section {{ background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 24px 26px; margin-bottom: 18px; }}
  h2 {{ font-size: 18px; margin: 0 0 14px; color: #1E293B; }}
  p {{ line-height: 1.55; font-size: 14px; color: #334155; }}
  .summary {{ font-size: 15px; }}
  .insight {{ border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }}
  .insight-title {{ font-weight: 700; font-size: 14.5px; margin-bottom: 4px; }}
  .insight p {{ margin: 0; font-size: 13.5px; }}
  .insight .action {{ margin-top: 6px; color: #475569; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .chart {{ padding: 18px 20px; }}
  .table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; display: block; overflow-x: auto; }}
  .table th {{ background: #F1F5F9; text-align: left; padding: 9px 10px; position: sticky; top: 0; }}
  .table td {{ border-top: 1px solid #E2E8F0; padding: 8px 10px; }}
  .table tr:nth-child(even) td {{ background: #FAFBFD; }}
  footer {{ text-align: center; color: #64748B; font-size: 12px; padding: 18px 0 8px; }}
  @media print {{
    body {{ background: white; padding: 0; }}
    header, section, .kpi {{ break-inside: avoid; }}
    section {{ border: 1px solid #CBD5E1; }}
  }}
  @media (max-width: 760px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Informe de datos y hallazgos</h1>
    <p>{html.escape(dataset_name)} · generado el {generated}</p>
    <span class="badge">DataLimpia Analytics · calidad {quality['puntaje']:.0f}/100 ({quality_word})</span>
  </header>

  <div class="kpis">{kpis}</div>

  <section>
    <h2>Resumen ejecutivo</h2>
    <p class="summary">{html.escape(summary_text)}</p>
  </section>

  <section>
    <h2>Hallazgos y acciones recomendadas</h2>
    {insights_html}
  </section>

  {''.join(charts_html)}

  <section>
    <h2>Perfil de columnas (primeras 40)</h2>
    {profile_html}
  </section>

  <section>
    <h2>Bitácora de limpieza</h2>
    {log_html}
  </section>

  <footer>Generado automáticamente con DataLimpia Analytics (Python + Streamlit).<br>
  Las correlaciones no implican causalidad; valida los hallazgos con el negocio.</footer>
</div>
</body>
</html>"""
