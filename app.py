"""DataLimpia Analytics · app principal de Streamlit."""

from __future__ import annotations

import html as html_lib
import itertools
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.charts import (
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
from src.cleaning import (
    CleaningConfig,
    CleaningResult,
    apply_cleaning,
    detect_sensitive_columns,
)
from src.config import APP_NAME, APP_TAGLINE, APP_VERSION
from src.insights import executive_summary, generate_insights
from src.io_utils import list_excel_sheets, load_dataset
from src.profiling import (
    column_roles,
    dataset_summary,
    detect_business_metrics,
    diagnose,
    profile_columns,
    quality_components,
    quality_label,
)
from src.report import build_csv, build_excel, build_html_report
from src.sample_data import build_sample_dataset
from src.utils import (
    categorical_columns,
    date_columns,
    fmt_number,
    fmt_pct,
    human_bytes,
    is_numeric_col,
    numeric_columns,
    safe_filename,
    text_columns,
    truncate,
)

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} · Análisis de datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_LOAD = "📁  Cargar datos"
PAGE_CLEAN = "🧹  Limpieza y calidad"
PAGE_EXPLORE = "🔎  Explorar"
PAGE_DASH = "📊  Dashboard"
PAGE_REPORT = "📄  Informe"
PAGES = [PAGE_LOAD, PAGE_CLEAN, PAGE_EXPLORE, PAGE_DASH, PAGE_REPORT]
NO_FILTER = "(ninguna)"
NO_DATE = "(sin fecha)"

FREQ_OPTIONS = {
    "Diario": "D",
    "Semanal": "W",
    "Mensual": "MS",
    "Trimestral": "QS",
    "Anual": "YS",
}
AGG_OPTIONS = {"Suma": "sum", "Promedio": "mean", "Mediana": "median", "Conteo": "count"}
CASE_OPTIONS = {
    "Sin cambios": "none",
    "minúsculas": "lower",
    "MAYÚSCULAS": "upper",
    "Tipo Título": "title",
}
NUM_STRATEGIES = {
    "Mediana (recomendado)": "median",
    "Media": "mean",
    "Cero": "zero",
    "Valor anterior": "ffill",
    "Eliminar filas con nulos": "drop_rows",
    "Eliminar columnas con nulos": "drop_cols",
    "No imputar": "none",
}
CAT_STRATEGIES = {
    "Moda (recomendado)": "mode",
    "Valor constante": "constant",
    "Valor anterior": "ffill",
    "Eliminar filas con nulos": "drop_rows",
    "Eliminar columnas con nulos": "drop_cols",
    "No imputar": "none",
}
DECIMAL_OPTIONS = {
    "Automático": "auto",
    "1.234,56 (coma decimal)": "comma",
    "1,234.56 (punto decimal)": "dot",
}
OUTLIER_METHODS = {"Ninguno": "none", "Rango intercuartílico (IQR)": "iqr", "Z-score": "zscore"}
OUTLIER_ACTIONS = {"Recortar al límite": "clip", "Eliminar filas": "remove", "Marcar con columna": "flag"}

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "nav": PAGE_LOAD,
    "raw_df": None,
    "raw_name": "",
    "raw_bytes": None,
    "raw_meta": {},
    "upload_sig": None,
    "uploader_nonce": 0,
    "clean_result": None,
    "report_html": None,
    "report_error": None,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def queue_widget_reset() -> None:
    st.session_state["_clear_widgets_pending"] = True


def apply_pending_reset() -> None:
    if st.session_state.pop("_clear_widgets_pending", False):
        for key in list(st.session_state.keys()):
            if key.startswith(("cfg_", "dash_", "expl_", "report_", "reload_")):
                st.session_state.pop(key, None)


def reset_data() -> None:
    for key in ("raw_df", "raw_name", "raw_bytes", "raw_meta", "upload_sig", "clean_result",
                "report_html", "report_error"):
        st.session_state[key] = _DEFAULTS.get(key)
    st.session_state["uploader_nonce"] = int(st.session_state.get("uploader_nonce", 0)) + 1
    st.session_state["_pending_nav"] = PAGE_LOAD
    queue_widget_reset()
    st.rerun()


def clear_derived() -> None:
    st.session_state["clean_result"] = None
    st.session_state["report_html"] = None
    st.session_state["report_error"] = None


def set_dataset(df: pd.DataFrame, name: str, data: bytes | None, meta: dict | None) -> None:
    queue_widget_reset()
    st.session_state["raw_df"] = df
    st.session_state["raw_name"] = name
    st.session_state["raw_bytes"] = data
    st.session_state["raw_meta"] = meta or {}
    st.session_state["clean_result"] = None
    st.session_state["report_html"] = None
    st.session_state["report_error"] = None


def get_raw() -> pd.DataFrame | None:
    return st.session_state.get("raw_df")


def get_clean() -> pd.DataFrame | None:
    result = st.session_state.get("clean_result")
    return result.df if isinstance(result, CleaningResult) else None


def active_df() -> pd.DataFrame | None:
    return get_clean() if get_clean() is not None else get_raw()


def go_to(page: str) -> None:
    st.session_state["_pending_nav"] = page
    st.rerun()


# ---------------------------------------------------------------------------
# Caché
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_sample() -> pd.DataFrame:
    return build_sample_dataset()


@st.cache_data(show_spinner=False)
def cached_sheets(data: bytes) -> list[str]:
    return list_excel_sheets(data)


@st.cache_data(show_spinner=False)
def cached_diagnose(df: pd.DataFrame) -> pd.DataFrame:
    return diagnose(df)


@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame) -> pd.DataFrame:
    return profile_columns(df)


@st.cache_data(show_spinner=False)
def cached_insights(df: pd.DataFrame, stats: dict) -> list[dict]:
    return generate_insights(df, stats)


@st.cache_data(show_spinner=False)
def cached_excel(df: pd.DataFrame, log: pd.DataFrame, stats: dict, name: str) -> bytes:
    return build_excel(df, log, stats, dataset_name=name)


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
.stApp { background: linear-gradient(180deg, #F6F8FC 0%, #F8FAFC 55%, #F6F8FC 100%); }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1500px; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] { background: #0B1220; border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] * { color: #E2E8F0; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #CBD5E1; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 9px 12px; border-radius: 10px; margin-bottom: 2px; transition: all .15s ease;
    border: 1px solid transparent;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(37,99,235,.26); border-color: rgba(96,165,250,.35);
}
[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight: 500; font-size: 14px; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1); }
.brand { display: flex; align-items: center; gap: 10px; padding: 6px 2px 18px 2px; }
.brand-logo {
    width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 22px;
    background: linear-gradient(135deg, #2563EB, #0EA5E9);
    box-shadow: 0 6px 18px rgba(37,99,235,.35);
}
.brand-name { font-size: 17px; font-weight: 700; color: #FFFFFF; letter-spacing: -.2px; }
.brand-sub { font-size: 11px; color: #94A3B8; letter-spacing: .12em; text-transform: uppercase; }
.side-label { font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: #64748B; font-weight: 700; margin: 4px 0 8px; }
.side-card { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); border-radius: 12px; padding: 12px 13px; margin-top: 6px; }
.side-card .name { font-weight: 600; font-size: 13px; color: #F1F5F9; word-break: break-all; }
.side-card .meta { font-size: 11.5px; color: #94A3B8; margin-top: 3px; }
.side-pill { display: inline-block; margin-top: 8px; padding: 3px 9px; border-radius: 999px; font-size: 10.5px; font-weight: 600; }
.page-head { margin: 4px 0 18px; }
.page-head h1 { font-size: 30px; margin: 0 0 4px; letter-spacing: -.03em; color: #0F172A; }
.page-head p { margin: 0; color: #64748B; font-size: 14px; }
.hero {
    position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px 32px;
    background: radial-gradient(900px 380px at 8% -30%, rgba(59,130,246,.85) 0%, transparent 55%),
                linear-gradient(135deg, #0B1220 0%, #162A55 55%, #0E7490 100%);
    color: white; margin-bottom: 20px; box-shadow: 0 18px 40px rgba(15,23,42,.18);
}
.hero h1 { font-size: 34px; line-height: 1.12; margin: 12px 0 10px; letter-spacing: -.03em; max-width: 840px; }
.hero p { color: #DBEAFE; font-size: 15px; max-width: 760px; margin: 0; line-height: 1.55; }
.hero-badge {
    display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: 11px;
    letter-spacing: .12em; text-transform: uppercase; font-weight: 700;
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); color: #E0F2FE;
}
.hero-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; }
.hero-chip { background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18); border-radius: 9px; padding: 6px 11px; font-size: 12px; color: #E2E8F0; }
.kpi { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 15px; padding: 15px 16px; height: 100%; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.kpi .label { font-size: 10.5px; letter-spacing: .09em; text-transform: uppercase; color: #64748B; font-weight: 700; }
.kpi .value { font-size: 25px; font-weight: 700; color: #0F172A; margin-top: 5px; line-height: 1.15; letter-spacing: -.02em; }
.kpi .hint { font-size: 11.5px; color: #94A3B8; margin-top: 4px; }
.kpi .hint.good { color: #059669; }
.kpi .hint.bad { color: #DC2626; }
.score-card { display: flex; align-items: center; gap: 16px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 15px; padding: 14px 16px; height: 100%; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.ring { width: 66px; height: 66px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex: 0 0 66px; }
.ring-inner { width: 50px; height: 50px; border-radius: 50%; background: #FFFFFF; display: flex; align-items: center; justify-content: center; font-size: 17px; font-weight: 700; color: #0F172A; }
.score-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em; color: #64748B; font-weight: 700; }
.score-value { font-size: 18px; font-weight: 700; margin-top: 2px; }
.score-sub { font-size: 11.5px; color: #94A3B8; margin-top: 2px; }
.insight-card { border-radius: 13px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid #E2E8F0; border-left-width: 4px; background: #FFFFFF; }
.insight-card .t { font-weight: 700; font-size: 13.5px; margin-bottom: 3px; }
.insight-card .d { font-size: 12.5px; color: #475569; line-height: 1.5; }
.insight-card .a { font-size: 12px; color: #334155; margin-top: 6px; }
.insight-card.success { background: #F0FDF9; border-left-color: #10B981; }
.insight-card.success .t { color: #047857; }
.insight-card.warning { background: #FFFBEB; border-left-color: #F59E0B; }
.insight-card.warning .t { color: #B45309; }
.insight-card.danger { background: #FEF2F2; border-left-color: #EF4444; }
.insight-card.danger .t { color: #B91C1C; }
.insight-card.info { background: #EFF6FF; border-left-color: #2563EB; }
.insight-card.info .t { color: #1D4ED8; }
.empty { border: 2px dashed #CBD5E1; border-radius: 18px; padding: 40px 24px; text-align: center; background: #FFFFFF; }
.empty .icon { font-size: 34px; }
.empty h3 { margin: 8px 0 4px; color: #1E293B; }
.empty p { margin: 0; color: #64748B; font-size: 13.5px; }
.soft-note { background: #EFF6FF; border: 1px solid #DBEAFE; border-radius: 12px; padding: 11px 14px; color: #1E40AF; font-size: 12.5px; }
.warn-note { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 12px; padding: 11px 14px; color: #92400E; font-size: 12.5px; }
.mini-title { font-size: 12px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: #64748B; margin: 6px 0 8px; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 12px 14px; }
[data-testid="stMetricLabel"] { color: #64748B; }
[data-testid="stMetricValue"] { font-size: 23px; }
.stButton > button { border-radius: 10px; font-weight: 600; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #2563EB, #0EA5E9); border: 0; box-shadow: 0 4px 14px rgba(37,99,235,.28); }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.08); color: #E2E8F0; border: 1px solid rgba(255,255,255,.16);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,.16); color: #FFFFFF; border-color: rgba(255,255,255,.3);
}
.stDownloadButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #CBD5E1; border-radius: 14px; background: #F8FAFC; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 6px; }
[data-testid="stTabs"] button[data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 7px 14px; }
[data-testid="stExpander"] { background: #FFFFFF; border-radius: 13px; border: 1px solid #E2E8F0; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #E2E8F0; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 15px; }
hr { border-color: #E2E8F0; }
.small { font-size: 12px; color: #64748B; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Componentes de interfaz
# ---------------------------------------------------------------------------
def page_head(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-head"><h1>{html_lib.escape(title)}</h1>'
        f"<p>{html_lib.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, hint: str = "", hint_class: str = "") -> None:
    hint_html = f'<div class="hint {hint_class}">{html_lib.escape(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="kpi"><div class="label">{html_lib.escape(label)}</div>'
        f'<div class="value">{html_lib.escape(value)}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def display_frame(data, **kwargs) -> None:
    """Muestra tablas convirtiendo columnas con tipos mezclados a texto.

    Evita que Arrow falle al serializar columnas que combinan números y texto
    (algo habitual en archivos recién cargados).
    """
    if not isinstance(data, pd.DataFrame):
        st.dataframe(data, **kwargs)
        return
    safe = data.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            values = safe[col].dropna()
            if not values.empty and values.map(type).nunique() > 1:
                safe[col] = safe[col].map(
                    lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
                )
    st.dataframe(safe, **kwargs)


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=key or f"plot_{next(_PLOT_COUNTER)}",
    )


def insight_cards(insights: list[dict], limit: int | None = None) -> None:
    for insight in insights[:limit] if limit else insights:
        action = insight.get("accion", "")
        action_html = (
            f'<div class="a"><strong>Acción:</strong> {html_lib.escape(str(action))}</div>'
            if action
            else ""
        )
        st.markdown(
            f'<div class="insight-card {html_lib.escape(str(insight.get("nivel", "info")))}">'
            f'<div class="t">{html_lib.escape(str(insight.get("titulo", "")))}</div>'
            f'<div class="d">{html_lib.escape(str(insight.get("detalle", "")))}</div>'
            f"{action_html}</div>",
            unsafe_allow_html=True,
        )


def quality_card(score: float, compact: bool = False) -> None:
    label, icon = quality_label(score)
    color = "#059669" if score >= 90 else "#D97706" if score >= 75 else "#EA580C" if score >= 55 else "#DC2626"
    st.markdown(
        f'<div class="score-card">'
        f'<div class="ring" style="background: conic-gradient({color} {score * 3.6:.1f}deg, #E2E8F0 0deg);">'
        f'<div class="ring-inner">{score:.0f}</div></div>'
        f'<div><div class="score-label">Calidad de datos</div>'
        f'<div class="score-value" style="color:{color}">{icon} {label}</div>'
        f'<div class="score-sub">Puntaje sobre 100</div></div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, text: str, icon: str = "📂", cta: str | None = None) -> None:
    st.markdown(
        f'<div class="empty"><div class="icon">{icon}</div><h3>{html_lib.escape(title)}</h3>'
        f"<p>{html_lib.escape(text)}</p></div>",
        unsafe_allow_html=True,
    )
    if cta:
        st.write("")
        _, center, _ = st.columns([1, 1, 1])
        with center:
            if st.button(cta, type="primary", width="stretch"):
                go_to(PAGE_LOAD)


def _ensure_state(key: str, options: list, default=None) -> None:
    if not options:
        return
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def _ensure_multi(key: str, options: list, default: list) -> None:
    if key not in st.session_state:
        st.session_state[key] = [x for x in default if x in options]
    else:
        valid = [x for x in st.session_state[key] if x in options]
        st.session_state[key] = valid


def _source_df() -> pd.DataFrame | None:
    clean = get_clean()
    if clean is not None:
        return clean
    raw = get_raw()
    if raw is None:
        return None
    return raw.drop(columns=detect_sensitive_columns(raw), errors="ignore")


_PLOT_COUNTER = itertools.count()


# ---------------------------------------------------------------------------
# Página 1 · Carga
# ---------------------------------------------------------------------------
def handle_upload(uploaded, *, force: bool = False, encoding: str = "auto",
                  separator: str = "auto", sheet: str | None = None) -> None:
    if uploaded is None:
        return
    signature = (uploaded.name, uploaded.size, encoding, separator, str(sheet))
    if not force and st.session_state.get("upload_sig") == signature:
        return
    st.session_state["upload_sig"] = signature
    data = uploaded.getvalue()
    try:
        with st.spinner("Leyendo y validando el archivo..."):
            result = load_dataset(
                data,
                uploaded.name,
                excel_sheet=sheet if sheet not in (None, "") else 0,
                encoding=None if encoding == "auto" else encoding,
                separator=None if separator == "auto" else separator,
            )
    except Exception as exc:
        st.error(f"No pudimos leer **{uploaded.name}**: {exc}")
        st.info("Prueba cambiando la codificación o el separador en «Opciones de lectura».")
        return
    set_dataset(
        result.df,
        uploaded.name,
        data,
        {"encoding": result.encoding, "separator": result.separator, "sheet": result.sheet},
    )
    st.rerun()


def render_load() -> None:
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-badge">DataLimpia Analytics · v{APP_VERSION}</div>
  <h1>De datos desordenados a decisiones claras.</h1>
  <p>Sube un CSV, Excel, JSON o Parquet. La aplicación detecta problemas, limpia el dataset,
  calcula indicadores y genera un tablero con las gráficas que importan para tu negocio.</p>
  <div class="hero-chips">
    <span class="hero-chip">🧹 Limpieza trazable</span>
    <span class="hero-chip">🔎 Perfilado automático</span>
    <span class="hero-chip">📊 Gráficas para decidir</span>
    <span class="hero-chip">📄 Informe exportable</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    raw = get_raw()
    if raw is None:
        nonce = int(st.session_state.get("uploader_nonce", 0))
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("#### 1 · Sube tu dataset")
                st.caption("Formatos: CSV, Excel (.xlsx/.xls), JSON y Parquet. Máximo 300 MB.")
                uploaded = st.file_uploader(
                    "Arrastra el archivo aquí o haz clic para seleccionarlo",
                    type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls", "json", "parquet"],
                    key=f"uploader_main_{nonce}",
                    label_visibility="collapsed",
                )
                if uploaded is not None:
                    handle_upload(uploaded)
                st.caption("Los datos se procesan en tu equipo; no se envían a servicios externos.")
                if uploaded is not None:
                    with st.expander("Opciones de lectura avanzadas"):
                        st.caption("Si el archivo no se lee bien, ajusta la codificación, el separador o la hoja.")
                        oc1, oc2 = st.columns(2)
                        with oc1:
                            load_enc = st.selectbox(
                                "Codificación", ["auto", "utf-8", "utf-8-sig", "cp1252", "latin-1"], key="load_enc"
                            )
                        with oc2:
                            load_sep_label = st.selectbox(
                                "Separador",
                                ["Automático", "Coma (,)", "Punto y coma (;)", "Tabulador", "Barra vertical (|)"],
                                key="load_sep",
                            )
                        sep_map = {
                            "Automático": "auto",
                            "Coma (,)": ",",
                            "Punto y coma (;)": ";",
                            "Tabulador": "\t",
                            "Barra vertical (|)": "|",
                        }
                        load_sheet = None
                        if Path(uploaded.name).suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
                            sheets = cached_sheets(uploaded.getvalue())
                            if sheets:
                                _ensure_state("load_sheet", sheets, sheets[0])
                                load_sheet = st.selectbox("Hoja del libro", sheets, key="load_sheet")
                        if st.button("Cargar con estas opciones", type="primary", width="stretch"):
                            handle_upload(
                                uploaded,
                                force=True,
                                encoding=load_enc,
                                separator=sep_map[load_sep_label],
                                sheet=load_sheet,
                            )
            st.write("")
            demo_col, or_col = st.columns([1.2, 1])
            with demo_col:
                if st.button("Probar con un dataset de demostración", type="primary", width="stretch"):
                    df = cached_sample()
                    set_dataset(
                        df,
                        "ventas_demo.csv",
                        build_csv(df),
                        {"encoding": "utf-8", "separator": ",", "sheet": None},
                    )
                    st.rerun()
            with or_col:
                st.markdown(
                    '<div class="small" style="padding-top:10px;">Ideal para ver la herramienta '
                    "funcionando con 1.800 ventas de ejemplo.</div>",
                    unsafe_allow_html=True,
                )
        with right:
            with st.container(border=True):
                st.markdown("#### Cómo funciona")
                steps = [
                    ("1", "Carga", "Detectamos el formato, la codificación y los tipos de dato."),
                    ("2", "Diagnostica", "Revisamos nulos, duplicados, texto sucio y atípicos."),
                    ("3", "Limpia", "Aplicas las reglas con una bitácora de cada cambio."),
                    ("4", "Analiza", "Tablero automático con métricas, tendencias y Pareto."),
                    ("5", "Comparte", "Exporta CSV, Excel y un informe HTML autocontenido."),
                ]
                rows = "".join(
                    f'<div style="display:flex;gap:10px;padding:7px 0;">'
                    f'<div style="flex:0 0 24px;height:24px;border-radius:8px;background:#EFF6FF;color:#2563EB;'
                    f'font-weight:700;font-size:12px;display:flex;align-items:center;justify-content:center;">{n}</div>'
                    f'<div><div style="font-weight:600;font-size:13px;color:#1E293B;">{t}</div>'
                    f'<div style="font-size:12px;color:#64748B;">{d}</div></div></div>'
                    for n, t, d in steps
                )
                st.markdown(rows, unsafe_allow_html=True)
        return

    # Dataset ya cargado ------------------------------------------------------
    summary = dataset_summary(raw)
    quality = quality_components(raw)
    st.success(f"Dataset cargado: **{st.session_state['raw_name']}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Filas", fmt_number(summary["filas"], 0))
    with c2:
        kpi_card("Columnas", fmt_number(summary["columnas"], 0))
    with c3:
        kpi_card("Celdas vacías", fmt_pct(summary["nulos_pct"]), f"{summary['nulos']:,} celdas".replace(",", "."))
    with c4:
        kpi_card("Duplicados", fmt_number(summary["duplicados"], 0), fmt_pct(summary["duplicados_pct"]))
    with c5:
        kpi_card("Memoria", human_bytes(summary["memoria_mb"] * 1024 * 1024))

    st.write("")
    tabs = st.tabs(["👁️ Vista previa", "🧬 Tipos y perfil", "⚙️ Opciones de lectura"])
    with tabs[0]:
        safe_preview = raw.drop(columns=detect_sensitive_columns(raw), errors="ignore")
        display_frame(safe_preview.head(300), width="stretch", height=380, hide_index=True)
        hidden_count = raw.shape[1] - safe_preview.shape[1]
        if hidden_count:
            st.caption(f"Se ocultaron {hidden_count} columnas de credenciales en la vista previa.")
        if len(raw) > 300:
            st.caption(f"Mostrando 300 de {len(raw):,} filas. El dataset completo se usa en el análisis.")
    with tabs[1]:
        profile = cached_profile(
            raw.drop(columns=detect_sensitive_columns(raw), errors="ignore")
        )
        display_frame(profile, width="stretch", height=380, hide_index=True)
        meta = st.session_state.get("raw_meta") or {}
        bits = []
        if meta.get("encoding"):
            bits.append(f"Codificación: `{meta['encoding']}`")
        if meta.get("separator"):
            sep = {"\t": "tabulador", ",": "coma", ";": "punto y coma", "|": "barra vertical"}.get(
                meta["separator"], meta["separator"]
            )
            bits.append(f"Separador: `{sep}`")
        if meta.get("sheet"):
            bits.append(f"Hoja: `{meta['sheet']}`")
        if bits:
            st.caption(" · ".join(bits))
    with tabs[2]:
        nonce = int(st.session_state.get("uploader_nonce", 0))
        uploaded = st.file_uploader(
            "Selecciona el archivo de nuevo para cambiar las opciones",
            type=["csv", "txt", "tsv", "xlsx", "xlsm", "xls", "json", "parquet"],
            key=f"uploader_reload_{nonce}",
        )
        if uploaded is None and st.session_state.get("raw_bytes") is not None:
            st.caption("Puedes volver a subir el archivo para ajustar codificación, separador u hoja.")
        if uploaded is not None:
            enc_col, sep_col = st.columns(2)
            with enc_col:
                encoding = st.selectbox(
                    "Codificación", ["auto", "utf-8", "utf-8-sig", "cp1252", "latin-1"], key="reload_encoding"
                )
            with sep_col:
                sep_label = st.selectbox(
                    "Separador",
                    ["Automático", "Coma (,)", "Punto y coma (;)", "Tabulador", "Barra vertical (|)"],
                    key="reload_sep",
                )
            sep_map = {"Automático": "auto", "Coma (,)": ",", "Punto y coma (;)": ";",
                       "Tabulador": "\t", "Barra vertical (|)": "|"}
            sheet = None
            if Path(uploaded.name).suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
                sheets = cached_sheets(uploaded.getvalue())
                if sheets:
                    sheet = st.selectbox("Hoja del libro", sheets, key="reload_sheet")
            if st.button("Recargar con estas opciones", type="primary"):
                handle_upload(
                    uploaded,
                    force=True,
                    encoding=encoding,
                    separator=sep_map[sep_label],
                    sheet=sheet,
                )

    st.write("")
    cta1, cta2, _ = st.columns([1.1, 1, 2.2])
    with cta1:
        if st.button("Continuar a la limpieza →", type="primary", width="stretch"):
            go_to(PAGE_CLEAN)
    with cta2:
        if st.button("Empezar de nuevo", width="stretch"):
            reset_data()


# ---------------------------------------------------------------------------
# Página 2 · Limpieza
# ---------------------------------------------------------------------------
def build_config() -> CleaningConfig:
    numeric_cols = numeric_columns(get_raw()) if get_raw() is not None else []
    return CleaningConfig(
        normalize_columns=st.session_state.get("cfg_norm", True),
        drop_empty_rows=st.session_state.get("cfg_drop_empty_rows", True),
        drop_empty_cols=st.session_state.get("cfg_drop_empty_cols", True),
        drop_constant_cols=st.session_state.get("cfg_drop_const", True),
        drop_high_null_cols=st.session_state.get("cfg_drop_high", False),
        high_null_threshold=st.session_state.get("cfg_threshold", 0.8),
        drop_zero_cols=st.session_state.get("cfg_drop_zero", True),
        zero_dominance_threshold=st.session_state.get("cfg_zero_threshold", 90) / 100,
        drop_sensitive_cols=st.session_state.get("cfg_drop_sensitive", True),
        sensitive_cols=st.session_state.get("cfg_sensitive_cols", []),
        clean_text=st.session_state.get("cfg_text", True),
        remove_accents=st.session_state.get("cfg_accents", False),
        case_mode=CASE_OPTIONS.get(st.session_state.get("cfg_case", "Sin cambios"), "none"),
        replace_placeholders=st.session_state.get("cfg_placeholders", True),
        auto_types=st.session_state.get("cfg_autotypes", True),
        decimal_mode=DECIMAL_OPTIONS.get(st.session_state.get("cfg_decimal", "Automático"), "auto"),
        dayfirst=st.session_state.get("cfg_dayfirst", True),
        force_numeric_cols=st.session_state.get("cfg_force_num", []),
        skip_numeric_cols=st.session_state.get("cfg_skip_num", []),
        force_date_cols=st.session_state.get("cfg_force_date", []),
        skip_date_cols=st.session_state.get("cfg_skip_date", []),
        drop_duplicates=st.session_state.get("cfg_dupes", True),
        duplicate_subset=st.session_state.get("cfg_dupe_subset", []) or None,
        numeric_strategy=NUM_STRATEGIES.get(
            st.session_state.get("cfg_num_strategy", "Mediana (recomendado)"), "median"
        ),
        categorical_strategy=CAT_STRATEGIES.get(
            st.session_state.get("cfg_cat_strategy", "Moda (recomendado)"), "mode"
        ),
        constant_value=st.session_state.get("cfg_constant", "Desconocido"),
        impute_exclude=st.session_state.get("cfg_impute_exclude", []),
        outlier_method=OUTLIER_METHODS.get(st.session_state.get("cfg_out_method", "Rango intercuartílico (IQR)"), "iqr"),
        outlier_action=OUTLIER_ACTIONS.get(st.session_state.get("cfg_out_action", "Recortar al límite"), "clip"),
        outlier_factor=st.session_state.get("cfg_out_factor", 1.5),
        outlier_cols=st.session_state.get("cfg_out_cols", numeric_cols),
    )


def render_cleaning() -> None:
    raw = get_raw()
    if raw is None:
        page_head("Limpieza y calidad", "Diagnostica y corrige los problemas del dataset antes de analizarlo.")
        empty_state(
            "Todavía no hay datos",
            "Carga un archivo o prueba el dataset de demostración para comenzar.",
            "🧹",
            "Ir a cargar datos",
        )
        return

    result: CleaningResult | None = st.session_state.get("clean_result")
    current = result.df if result is not None else raw
    detected_sensitive = detect_sensitive_columns(raw)
    page_head(
        "Limpieza y calidad",
        "Aplica reglas de limpieza con trazabilidad completa. Cada cambio queda en la bitácora.",
    )
    if detected_sensitive:
        st.info(
            f"Detectamos {len(detected_sensitive)} columnas de credenciales por sus nombres. "
            "Se ocultan en las vistas del dataset original y se eliminan al limpiar por defecto."
        )

    summary = dataset_summary(current)
    quality = quality_components(current)
    qrow = st.columns([1.25, 1, 1, 1, 1])
    with qrow[0]:
        quality_card(quality["puntaje"])
    with qrow[1]:
        kpi_card("Filas", fmt_number(summary["filas"], 0), "después de limpiar" if result else "dataset original")
    with qrow[2]:
        kpi_card("Columnas", fmt_number(summary["columnas"], 0))
    with qrow[3]:
        kpi_card("Celdas vacías", fmt_pct(summary["nulos_pct"]), fmt_number(summary["nulos"], 0) + " celdas")
    with qrow[4]:
        kpi_card("Duplicados", fmt_number(summary["duplicados"], 0), fmt_pct(summary["duplicados_pct"]))

    st.write("")
    diag_tab, profile_tab, missing_tab = st.tabs(
        ["🩺 Diagnóstico", "🧬 Perfil de columnas", "🕳️ Valores nulos"]
    )
    with diag_tab:
        diag = cached_diagnose(raw)
        show = diag.copy()
        icons = {"Alta": "🔴 Alta", "Media": "🟠 Media", "Baja": "🟡 Baja"}
        show["Severidad"] = show["Severidad"].map(icons).fillna(show["Severidad"])
        display_frame(show, width="stretch", hide_index=True, height=min(430, 80 + 36 * len(show)))
        st.caption("El diagnóstico se calcula sobre el dataset original, antes de limpiar.")
    with profile_tab:
        profile_source = current.drop(
            columns=detect_sensitive_columns(current), errors="ignore"
        )
        display_frame(cached_profile(profile_source), width="stretch", hide_index=True, height=420)
    with missing_tab:
        plot(missing_values_chart(current))
        st.caption("Las columnas sin barra no tienen valores nulos.")

    st.write("")
    with st.expander("⚙️ Configurar la limpieza", expanded=result is None):
        cfg_tabs = st.tabs(["Estructura y texto", "Tipos de datos", "Valores nulos", "Duplicados y atípicos"])
        with cfg_tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.toggle("Normalizar nombres de columnas (snake_case)", value=True, key="cfg_norm")
                st.toggle("Limpiar espacios en textos", value=True, key="cfg_text")
                st.selectbox("Formato del texto", list(CASE_OPTIONS), key="cfg_case")
                st.toggle("Quitar acentos", value=False, key="cfg_accents")
                st.toggle("Convertir 'N/A', '-', 'sin dato' en nulos", value=True, key="cfg_placeholders")
            with c2:
                st.toggle("Eliminar filas vacías", value=True, key="cfg_drop_empty_rows")
                st.toggle("Eliminar columnas vacías", value=True, key="cfg_drop_empty_cols")
                st.toggle("Eliminar columnas constantes", value=True, key="cfg_drop_const")
                st.toggle(
                    "Eliminar columnas binarias dominadas por 0",
                    value=True,
                    key="cfg_drop_zero",
                    help="Elimina indicadores 0/1 con al menos el porcentaje de ceros elegido. "
                    "Las columnas con otros valores no se eliminan por esta regla.",
                )
                st.slider(
                    "Porcentaje mínimo de ceros en columnas binarias",
                    min_value=90,
                    max_value=99,
                    value=90,
                    step=1,
                    format="%d%%",
                    key="cfg_zero_threshold",
                    disabled=not st.session_state.get("cfg_drop_zero", True),
                )
                st.toggle(
                    "Eliminar columnas de credenciales detectadas",
                    value=True,
                    key="cfg_drop_sensitive",
                )
                _ensure_multi(
                    "cfg_sensitive_cols",
                    list(raw.columns),
                    detected_sensitive,
                )
                st.multiselect(
                    "Columnas de credenciales a eliminar",
                    list(raw.columns),
                    key="cfg_sensitive_cols",
                    help="La detección usa solo nombres de columnas; nunca inspecciona el contenido.",
                )
                drop_high = st.toggle("Eliminar columnas con muchos nulos", value=False, key="cfg_drop_high")
                st.slider(
                    "Umbral de nulos para eliminar la columna",
                    0.5, 1.0, 0.8, 0.05,
                    key="cfg_threshold",
                    disabled=not drop_high,
                    help="Por ejemplo, 0,80 elimina columnas con 80% o más de valores vacíos.",
                )
        with cfg_tabs[1]:
            st.toggle("Detectar y convertir tipos automáticamente", value=True, key="cfg_autotypes")
            c1, c2 = st.columns(2)
            with c1:
                st.selectbox(
                    "Interpretación de números",
                    list(DECIMAL_OPTIONS),
                    key="cfg_decimal",
                    help="Define si '1.234' es mil doscientos treinta y cuatro o uno con decimales.",
                )
                st.toggle(
                    "Fechas en formato día/mes/año",
                    value=True,
                    key="cfg_dayfirst",
                    help="Desactívalo si tus fechas están en formato mes/día/año.",
                )
            with c2:
                txt_cols = text_columns(raw)
                st.multiselect("Forzar columnas a número", txt_cols, key="cfg_force_num")
                st.multiselect("Forzar columnas a fecha", txt_cols, key="cfg_force_date")
                st.multiselect("Excluir de conversión numérica", txt_cols, key="cfg_skip_num")
                st.multiselect("Excluir de conversión de fecha", txt_cols, key="cfg_skip_date")
        with cfg_tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                st.selectbox("Estrategia para columnas numéricas", list(NUM_STRATEGIES), key="cfg_num_strategy")
                st.text_input(
                    "Valor para nulos categóricos (si eliges 'constante')",
                    value="Desconocido",
                    key="cfg_constant",
                )
            with c2:
                st.selectbox("Estrategia para columnas categóricas", list(CAT_STRATEGIES), key="cfg_cat_strategy")
                st.multiselect(
                    "Columnas que no se deben imputar",
                    list(raw.columns),
                    key="cfg_impute_exclude",
                )
            st.markdown(
                '<div class="soft-note">Las estrategias de «eliminar filas/columnas» se aplican solo '
                "a columnas con nulos. Imputar la mediana es más seguro que el promedio cuando hay atípicos.</div>",
                unsafe_allow_html=True,
            )
        with cfg_tabs[3]:
            c1, c2 = st.columns(2)
            with c1:
                st.toggle("Eliminar filas duplicadas", value=True, key="cfg_dupes")
                st.multiselect(
                    "Columnas para detectar duplicados (vacío = todas)",
                    list(raw.columns),
                    key="cfg_dupe_subset",
                )
            with c2:
                st.selectbox("Método de atípicos", list(OUTLIER_METHODS), index=1, key="cfg_out_method")
                st.selectbox("Acción sobre atípicos", list(OUTLIER_ACTIONS), key="cfg_out_action")
                st.slider("Sensibilidad (factor IQR / Z)", 1.0, 5.0, 1.5, 0.1, key="cfg_out_factor")
                st.multiselect(
                    "Columnas numéricas a revisar",
                    numeric_columns(raw),
                    default=numeric_columns(raw),
                    key="cfg_out_cols",
                )

        st.write("")
        b1, b2, _ = st.columns([1.2, 1, 2.2])
        with b1:
            apply_clicked = st.button("✨ Aplicar limpieza", type="primary", width="stretch")
        with b2:
            reset_clicked = st.button("Restablecer reglas", width="stretch")
        if reset_clicked:
            clear_derived()
            queue_widget_reset()
            st.rerun()
        if apply_clicked:
            cfg = build_config()
            try:
                with st.spinner("Aplicando limpieza..."):
                    cleaned = apply_cleaning(raw, cfg)
                st.session_state["clean_result"] = cleaned
                st.session_state["report_html"] = None
                st.session_state["report_error"] = None
                st.success("Limpieza aplicada correctamente.")
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo aplicar la limpieza: {exc}")

    if result is None:
        st.markdown(
            '<div class="soft-note">Aún no has aplicado la limpieza. Revisa el diagnóstico y '
            "ajusta las reglas; los valores por defecto ya resuelven los problemas más comunes.</div>",
            unsafe_allow_html=True,
        )
        return

    st.write("")
    st.markdown('<div class="mini-title">Resultado de la limpieza</div>', unsafe_allow_html=True)
    s = result.stats
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card(
            "Filas",
            fmt_number(s["filas_despues"], 0),
            f"{fmt_number(s['filas_antes'], 0)} → {fmt_number(s['filas_despues'], 0)}",
        )
    with m2:
        kpi_card(
            "Columnas",
            fmt_number(s["columnas_despues"], 0),
            f"{fmt_number(s['columnas_antes'], 0)} → {fmt_number(s['columnas_despues'], 0)}",
        )
    with m3:
        kpi_card(
            "Nulos",
            fmt_pct(s["nulos_despues"]),
            f"antes {fmt_pct(s['nulos_antes'])}",
        )
    with m4:
        after_score = quality_components(result.df)["puntaje"]
        kpi_card(
            "Calidad",
            f"{after_score:.0f}/100",
            f"antes {quality_components(raw)['puntaje']:.0f}/100",
            "good" if after_score >= 75 else "bad",
        )

    st.write("")
    res_tabs = st.tabs(["🧾 Bitácora", "👁️ Datos limpios", "💾 Descargar"])
    with res_tabs[0]:
        display_frame(result.log, width="stretch", hide_index=True, height=min(420, 80 + 34 * len(result.log)))
    with res_tabs[1]:
        display_frame(result.df.head(500), width="stretch", height=420, hide_index=True)
        st.caption(f"{len(result.df):,} filas · {result.df.shape[1]} columnas".replace(",", "."))
    with res_tabs[2]:
        name = st.session_state.get("raw_name", "dataset.csv")
        stem = Path(name).stem
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Descargar CSV limpio",
                data=build_csv(result.df),
                file_name=safe_filename(f"{stem}_limpio", ".csv"),
                mime="text/csv",
                width="stretch",
            )
        with d2:
            st.download_button(
                "⬇️ Descargar Excel con informe",
                data=cached_excel(result.df, result.log, result.stats, name),
                file_name=safe_filename(f"{stem}_limpio", ".xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        st.caption("El Excel incluye los datos limpios, el resumen, el perfil y la bitácora de cambios.")

    st.write("")
    cta1, cta2, _ = st.columns([1.1, 1, 2.2])
    with cta1:
        if st.button("Explorar los datos →", type="primary", width="stretch"):
            go_to(PAGE_EXPLORE)
    with cta2:
        if st.button("Ver dashboard →", width="stretch"):
            go_to(PAGE_DASH)


# ---------------------------------------------------------------------------
# Página 3 · Explorar
# ---------------------------------------------------------------------------
def render_explore() -> None:
    raw, clean = get_raw(), get_clean()
    if raw is None:
        page_head("Explorar", "Filtra, ordena y revisa el detalle de los registros.")
        empty_state("No hay datos para explorar", "Carga un dataset primero.", "🔎", "Ir a cargar datos")
        return

    page_head("Explorar", "Filtra los registros, revisa estadísticas y encuentra valores puntuales.")

    options = ["Datos limpios", "Datos originales"] if clean is not None else ["Datos originales"]
    _ensure_state("expl_source", options, options[0])
    source = st.radio("Fuente", options, horizontal=True, key="expl_source", label_visibility="collapsed")
    df = clean if source == "Datos limpios" and clean is not None else raw
    if source == "Datos originales":
        visible_sensitive = detect_sensitive_columns(df)
        if visible_sensitive:
            df = df.drop(columns=visible_sensitive, errors="ignore")
            st.caption(
                f"Se ocultaron {len(visible_sensitive)} columnas de credenciales de los datos "
                "originales, incluidos filtros y descargas."
            )

    cats_all = categorical_columns(df, max_unique=200)
    nums_all = numeric_columns(df)
    dates_all = date_columns(df)
    search_text = ""
    cat_col = NO_FILTER
    cat_values: list = []
    date_col = NO_DATE
    date_range = None
    num_col = NO_FILTER
    num_range = None

    with st.container(border=True):
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            search_text = st.text_input(
                "Buscar texto",
                key="expl_search",
                placeholder="Ej.: nombre de producto, ciudad, vendedor…",
            )
        with c2:
            cat_col = st.selectbox("Filtrar por categoría", [NO_FILTER] + cats_all, key="expl_cat")
        with c3:
            if cat_col != NO_FILTER and cat_col in df.columns:
                values = sorted(df[cat_col].dropna().astype(str).unique().tolist())
                cat_values = st.multiselect("Valores a incluir", values, key=f"expl_vals_{cat_col}")
            else:
                st.caption("Elige una categoría para filtrar por sus valores.")

        c4, c5, c6 = st.columns([1, 1.2, 1])
        with c4:
            date_col = st.selectbox("Filtrar por fecha", [NO_DATE] + dates_all, key="expl_date")
        with c5:
            if date_col != NO_DATE and date_col in df.columns:
                valid = df[date_col].dropna()
                if not valid.empty:
                    min_d = valid.min().date()
                    max_d = valid.max().date()
                    if min_d < max_d:
                        selected = st.date_input(
                            "Rango de fechas",
                            value=(min_d, max_d),
                            min_value=min_d,
                            max_value=max_d,
                            key=f"expl_dates_{source}_{date_col}",
                        )
                        if isinstance(selected, (tuple, list)) and len(selected) == 2:
                            date_range = selected
                        elif selected:
                            date_range = (selected, selected)
                    else:
                        st.caption(f"Todos los registros son del {min_d:%d/%m/%Y}.")
            else:
                st.caption("Opcional: limita el periodo analizado.")
        with c6:
            num_col = st.selectbox("Filtrar por rango numérico", [NO_FILTER] + nums_all, key="expl_num")
        if num_col != NO_FILTER and num_col in df.columns:
            valid = pd.to_numeric(df[num_col], errors="coerce").dropna()
            if not valid.empty:
                lo, hi = float(valid.min()), float(valid.max())
                if lo < hi:
                    num_range = st.slider(
                        f"Rango de {num_col}",
                        lo,
                        hi,
                        (lo, hi),
                        key=f"expl_range_{source}_{num_col}",
                    )

    mask = pd.Series(True, index=df.index)
    active_filters = 0
    if search_text.strip():
        query = search_text.strip()
        text_cols = text_columns(df)
        if text_cols:
            mask &= (
                df[text_cols]
                .astype(str)
                .apply(lambda s: s.str.contains(query, case=False, na=False, regex=False))
                .any(axis=1)
            )
            active_filters += 1
    if cat_col != NO_FILTER and cat_values:
        mask &= df[cat_col].astype(str).isin(cat_values)
        active_filters += 1
    if date_col != NO_DATE and date_range:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        mask &= df[date_col].between(start, end)
        active_filters += 1
    if num_col != NO_FILTER and num_range:
        mask &= pd.to_numeric(df[num_col], errors="coerce").between(num_range[0], num_range[1])
        active_filters += 1

    filtered = df[mask]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Registros mostrados", fmt_number(len(filtered), 0), f"de {fmt_number(len(df), 0)}")
    with c2:
        kpi_card("Proporción", fmt_pct(len(filtered) / max(len(df), 1)))
    with c3:
        kpi_card("Columnas", fmt_number(filtered.shape[1], 0))
    with c4:
        kpi_card("Filtros activos", str(active_filters), "Búsqueda, categoría, fecha o rango")

    if filtered.empty:
        st.warning("Ningún registro cumple los filtros seleccionados. Ajusta los criterios.")
        return

    st.write("")
    tabs = st.tabs(["📋 Tabla", "📈 Resumen estadístico", "🏷️ Frecuencias"])
    with tabs[0]:
        display_frame(filtered.head(2000), width="stretch", height=460, hide_index=True)
        if len(filtered) > 2000:
            st.caption(f"Vista limitada a 2.000 filas. Descarga el archivo para ver todas las {len(filtered):,}.")
        st.download_button(
            "⬇️ Descargar selección (CSV)",
            data=build_csv(filtered),
            file_name=safe_filename(f"{Path(st.session_state.get('raw_name', 'dataset')).stem}_filtrado", ".csv"),
            mime="text/csv",
        )
    with tabs[1]:
        display_frame(cached_profile(filtered), width="stretch", height=460, hide_index=True)
        if nums_all:
            with st.expander("Estadísticas descriptivas de variables numéricas"):
                desc = filtered[nums_all].apply(pd.to_numeric, errors="coerce").describe().T
                display_frame(desc, width="stretch")
    with tabs[2]:
        freq_cols = cats_all or text_columns(filtered)
        if not freq_cols:
            st.info("No hay columnas categóricas para analizar.")
        else:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                _ensure_state("expl_freq_col", freq_cols, freq_cols[0])
                freq_col = st.selectbox("Columna", freq_cols, key="expl_freq_col")
            with c2:
                top_n = st.slider("Top valores", 3, 30, 10, key="expl_freq_top")
            plot(category_bar(filtered, freq_col, top_n=top_n))
            counts = filtered[freq_col].astype(str).value_counts().head(top_n).rename_axis(freq_col).reset_index(name="Registros")
            counts["% del total"] = counts["Registros"] / len(filtered)
            display_frame(counts, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Página 4 · Dashboard
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    df = _source_df()
    if df is None:
        page_head("Dashboard", "Indicadores y gráficas para la toma de decisiones.")
        empty_state("No hay datos para graficar", "Carga un dataset para generar el tablero.", "📊", "Ir a cargar datos")
        return

    page_head("Dashboard", "Indicadores, tendencias y relaciones calculadas automáticamente sobre tus datos.")

    if get_clean() is None:
        st.markdown(
            '<div class="warn-note">Estás viendo los datos originales. '
            "Aplica la limpieza para evitar conclusiones basadas en nulos, duplicados o tipos incorrectos.</div>",
            unsafe_allow_html=True,
        )

    roles = detect_business_metrics(df)
    nums = numeric_columns(df)
    cats = categorical_columns(df, max_unique=60)
    dates = date_columns(df)
    if not nums:
        st.warning("No hay columnas numéricas para calcular métricas. Convierte alguna columna a número en la limpieza.")
        display_frame(df.head(100), width="stretch", height=360)
        return

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.1, 1, 0.9, 0.9])
        with c1:
            _ensure_state("dash_metric", nums, roles.get("metric"))
            metric = st.selectbox("Métrica principal", nums, key="dash_metric")
        with c2:
            dim_options = cats or [NO_FILTER]
            _ensure_state("dash_dim", dim_options, roles.get("dimension"))
            dimension = st.selectbox("Dimensión", dim_options, key="dash_dim")
            if dimension == NO_FILTER:
                dimension = None
        with c3:
            date_options = [NO_DATE] + dates
            _ensure_state("dash_date", date_options, roles.get("date"))
            date_choice = st.selectbox("Fecha", date_options, key="dash_date")
            date_col = None if date_choice == NO_DATE else date_choice
        with c4:
            _ensure_state("dash_freq", list(FREQ_OPTIONS), "Mensual")
            freq_label = st.selectbox("Periodo", list(FREQ_OPTIONS), key="dash_freq")
        with c5:
            _ensure_state("dash_agg", list(AGG_OPTIONS), "Suma")
            agg_label = st.selectbox("Agregación", list(AGG_OPTIONS), key="dash_agg")

    freq = FREQ_OPTIONS[freq_label]
    agg = AGG_OPTIONS[agg_label]
    metric_series = pd.to_numeric(df[metric], errors="coerce")
    total_metric = float(metric_series.sum())
    avg_metric = float(metric_series.mean()) if metric_series.notna().any() else 0.0
    quality = quality_components(df)

    top_name, top_share = "—", 0.0
    if dimension and dimension in df.columns:
        grouped = df.dropna(subset=[dimension]).groupby(dimension, observed=True)[metric].sum()
        if not grouped.empty and grouped.sum() != 0:
            top_name = str(grouped.idxmax())
            top_share = float(grouped.max() / grouped.sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Registros", fmt_number(len(df), 0), f"{df.shape[1]} columnas")
    with k2:
        kpi_card("Total métrica", fmt_number(total_metric), metric.replace("_", " "))
    with k3:
        kpi_card("Promedio", fmt_number(avg_metric), f"por registro")
    with k4:
        kpi_card("Líder", top_name, f"{fmt_pct(top_share)} del total" if dimension else "sin dimensión")
    with k5:
        kpi_card("Calidad", f"{quality['puntaje']:.0f}/100", f"completitud {fmt_pct(quality['completitud'])}")

    st.write("")
    tabs = st.tabs(["🧭 Resumen", "📈 Distribuciones", "🔗 Relaciones", "🏷️ Categorías", "🕒 Tiempo", "🎨 Constructor"])

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            if dimension:
                plot(category_bar(df, dimension, metric, agg, top_n=12))
            else:
                plot(numeric_distribution(df, metric))
        with right:
            if dimension:
                plot(donut_chart(df, dimension, top_n=7))
            else:
                st.info("Añade una columna categórica para ver la composición.")
        c1, c2 = st.columns(2)
        with c1:
            plot(missing_values_chart(df))
        with c2:
            if dimension:
                second_dim = next(
                    (
                        c
                        for c in cats
                        if c != dimension and 2 <= df[c].nunique(dropna=True) <= 40
                    ),
                    None,
                )
                path = [dimension] + ([second_dim] if second_dim else [])
                plot(treemap_chart(df, path, metric))
            else:
                plot(numeric_grid(df, nums[:4]))
        st.markdown('<div class="mini-title">Hallazgos automáticos</div>', unsafe_allow_html=True)
        insights = cached_insights(df, st.session_state["clean_result"].stats if get_clean() is not None else {})
        insight_cards(insights, limit=4)

    with tabs[1]:
        _ensure_multi("dash_numcols", nums, nums[:4])
        selected_nums = st.multiselect("Variables numéricas", nums, key="dash_numcols")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            if selected_nums:
                plot(numeric_grid(df, selected_nums[:6]))
        with c2:
            _ensure_state("dash_hist_col", nums, metric)
            hist_col = st.selectbox("Detalle de una variable", nums, key="dash_hist_col")
            bins = st.slider("Número de intervalos", 10, 100, 40, key="dash_bins")
            plot(numeric_distribution(df, hist_col, bins))
        if selected_nums:
            desc = df[selected_nums].apply(pd.to_numeric, errors="coerce").describe().T
            desc.insert(0, "Variable", desc.index)
            display_frame(desc, width="stretch", hide_index=True)

    with tabs[2]:
        plot(correlation_heatmap(df, nums))
        if len(nums) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                _ensure_state("dash_scatter_x", nums, nums[0])
                sx = st.selectbox("Eje X", nums, key="dash_scatter_x")
            with c2:
                _ensure_state("dash_scatter_y", nums, nums[1] if len(nums) > 1 else nums[0])
                sy = st.selectbox("Eje Y", nums, key="dash_scatter_y")
            with c3:
                color_options = [NO_FILTER] + cats
                _ensure_state("dash_scatter_color", color_options, dimension or NO_FILTER)
                sc = st.selectbox("Color", color_options, key="dash_scatter_color")
            plot(scatter_plot(df, sx, sy, None if sc == NO_FILTER else sc))
        if dimension:
            plot(box_by_category(df, dimension, metric))
        st.caption("La línea punteada es una tendencia lineal; una correlación no implica causalidad.")

    with tabs[3]:
        if not dimension:
            st.info("Selecciona una dimensión categórica en la barra superior para analizar categorías.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                plot(category_bar(df, dimension, metric, agg, top_n=15))
            with c2:
                plot(pareto_chart(df, dimension, metric, agg, top_n=12))
            grouped = (
                df.dropna(subset=[dimension])
                .groupby(dimension, observed=True)
                .agg(Registros=(metric, "size"), Metrica=(metric, "sum"))
                .sort_values("Metrica", ascending=False)
                .reset_index()
            )
            grouped["Participación"] = grouped["Metrica"] / grouped["Metrica"].sum() if grouped["Metrica"].sum() else 0
            grouped["Participación"] = grouped["Participación"].map(fmt_pct)
            display_frame(grouped, width="stretch", hide_index=True)

    with tabs[4]:
        if not date_col:
            st.info("No hay una columna de fecha disponible. Convierte una columna a fecha en la limpieza.")
        else:
            plot(time_series(df, date_col, metric, freq=freq, agg=agg, color=dimension))
            work = df.dropna(subset=[date_col]).copy()
            if not work.empty:
                try:
                    series = work.set_index(date_col)[metric].resample(freq).agg(agg).dropna()
                except (TypeError, ValueError):
                    series = pd.Series(dtype=float)
                if len(series) >= 2:
                    last, prev = series.iloc[-1], series.iloc[-2]
                    change = (last - prev) / abs(prev) if prev else None
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        kpi_card("Último periodo", fmt_number(float(last)), f"{series.index[-1]:%d/%m/%Y}")
                    with c2:
                        kpi_card(
                            "Variación",
                            fmt_pct(change) if change is not None else "—",
                            "vs. periodo anterior",
                            "good" if change and change > 0 else "bad" if change and change < 0 else "",
                        )
                    with c3:
                        kpi_card("Mejor periodo", fmt_number(float(series.max())), f"{series.idxmax():%d/%m/%Y}")
                    table = series.reset_index()
                    table.columns = ["Periodo", metric]
                    table["Periodo"] = table["Periodo"].dt.strftime("%d/%m/%Y")
                    display_frame(table.tail(24).iloc[::-1], width="stretch", hide_index=True)
                else:
                    st.caption("Se necesitan al menos dos periodos para calcular la variación.")

    with tabs[5]:
        c1, c2, c3 = st.columns(3)
        kinds = ["Barras", "Barras horizontales", "Líneas", "Área", "Dispersión", "Cajas", "Histograma", "Torta", "Dona"]
        with c1:
            kind = st.selectbox("Tipo de gráfica", kinds, key="build_kind")
            x_options = list(df.columns)
            _ensure_state("build_x", x_options, dimension or metric)
            x = st.selectbox("Eje X / categoría", x_options, key="build_x")
        with c2:
            y_options = [NO_FILTER] + nums
            _ensure_state("build_y", y_options, metric)
            y = st.selectbox("Métrica (Y)", y_options, key="build_y")
            color_options = [NO_FILTER] + cats
            _ensure_state("build_color", color_options, NO_FILTER)
            color = st.selectbox("Color", color_options, key="build_color")
        with c3:
            _ensure_state("build_agg", list(AGG_OPTIONS), "Suma")
            agg2 = st.selectbox("Agregación", list(AGG_OPTIONS), key="build_agg")
            top_n = st.slider("Top categorías", 3, 30, 12, key="build_top")
            bins = st.slider("Intervalos (histograma)", 10, 100, 30, key="build_bins")
        plot(
            custom_chart(
                df,
                kind,
                x,
                None if y == NO_FILTER else y,
                None if color == NO_FILTER else color,
                AGG_OPTIONS[agg2],
                top_n,
                bins,
            )
        )
        st.caption("Combina cualquier columna para responder preguntas específicas de tu área.")

    st.write("")
    if st.button("Ver informe y exportar →", type="primary"):
        go_to(PAGE_REPORT)


# ---------------------------------------------------------------------------
# Página 5 · Informe
# ---------------------------------------------------------------------------
def report_figures(df: pd.DataFrame) -> list[tuple[str, object]]:
    roles = detect_business_metrics(df)
    nums = numeric_columns(df)
    metric, dim, date_col = roles.get("metric"), roles.get("dimension"), roles.get("date")
    figures: list[tuple[str, object]] = []
    figures.append(("Calidad: valores nulos", missing_values_chart(df)))
    if nums:
        figures.append(("Distribución de variables", numeric_grid(df, nums[:6])))
    if len(nums) >= 2:
        figures.append(("Correlación entre variables", correlation_heatmap(df)))
    if dim and metric:
        figures.append(("Categorías con mayor aporte", category_bar(df, dim, metric)))
        figures.append(("Pareto: concentración por categoría", pareto_chart(df, dim, metric)))
    if date_col and metric:
        figures.append(("Evolución en el tiempo", time_series(df, date_col, metric, freq="MS")))
    return figures


def render_report() -> None:
    df = _source_df()
    if df is None:
        page_head("Informe", "Genera un documento con los hallazgos y las gráficas principales.")
        empty_state("No hay datos para el informe", "Carga y limpia un dataset primero.", "📄", "Ir a cargar datos")
        return

    page_head("Informe", "Resumen ejecutivo, hallazgos y exportaciones listas para compartir.")

    if get_clean() is None:
        st.markdown(
            '<div class="warn-note">El informe se generará con los datos originales. '
            "Para mejores resultados, aplica primero la limpieza.</div>",
            unsafe_allow_html=True,
        )

    result = st.session_state.get("clean_result")
    stats = result.stats if isinstance(result, CleaningResult) else {}
    log = result.log if isinstance(result, CleaningResult) else pd.DataFrame()
    name = st.session_state.get("raw_name", "dataset.csv")
    stem = Path(name).stem
    insights = cached_insights(df, stats)
    summary_text = executive_summary(df, stats)
    figures = report_figures(df)

    with st.container(border=True):
        st.markdown("#### Resumen ejecutivo")
        st.markdown(f"<p style='font-size:14.5px;color:#334155;line-height:1.6;'>{html_lib.escape(summary_text)}</p>",
                    unsafe_allow_html=True)
        s = dataset_summary(df)
        quality = quality_components(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Registros", fmt_number(s["filas"], 0))
        with c2:
            kpi_card("Columnas", fmt_number(s["columnas"], 0))
        with c3:
            kpi_card("Calidad", f"{quality['puntaje']:.0f}/100", quality_label(quality["puntaje"])[0])
        with c4:
            kpi_card("Completitud", fmt_pct(quality["completitud"]))
        with c5:
            kpi_card("Duplicados eliminados", fmt_number(stats.get("duplicados_eliminados", 0), 0))

    st.write("")
    st.markdown('<div class="mini-title">Hallazgos y acciones</div>', unsafe_allow_html=True)
    insight_cards(insights)

    st.write("")
    with st.container(border=True):
        st.markdown("#### Exportar")
        st.caption("Comparte los datos limpios, un Excel con formato o un informe HTML autocontenido (se puede imprimir a PDF).")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "⬇️ CSV limpio",
                data=build_csv(df),
                file_name=safe_filename(f"{stem}_limpio", ".csv"),
                mime="text/csv",
                width="stretch",
            )
        with e2:
            st.download_button(
                "⬇️ Excel con informe",
                data=cached_excel(df, log, stats, name),
                file_name=safe_filename(f"{stem}_informe", ".xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with e3:
            generate = st.button("📄 Generar informe HTML", type="primary", width="stretch")
        if generate:
            try:
                with st.spinner("Generando informe con gráficas..."):
                    html_report = build_html_report(
                        df=df,
                        dataset_name=name,
                        stats=stats,
                        log=log,
                        insights=insights,
                        figures=figures,
                        summary_text=summary_text,
                    )
                st.session_state["report_html"] = html_report
                st.session_state["report_error"] = None
                st.success("Informe generado. Ya puedes descargarlo.")
            except Exception as exc:
                st.session_state["report_error"] = str(exc)
                st.error(f"No se pudo generar el informe: {exc}")
        if st.session_state.get("report_html"):
            st.download_button(
                "⬇️ Descargar informe HTML",
                data=st.session_state["report_html"],
                file_name=safe_filename(f"{stem}_informe", ".html"),
                mime="text/html",
                type="primary",
                width="stretch",
            )
        if st.session_state.get("report_error"):
            st.caption(f"Último error: {st.session_state['report_error']}")

    st.write("")
    st.markdown('<div class="mini-title">Vista previa del informe</div>', unsafe_allow_html=True)
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for j, (title, fig) in enumerate(figures[i:i + 2]):
            with cols[j]:
                st.markdown(f"**{title}**")
                plot(fig)
    with st.expander("Ver bitácora de limpieza"):
        if isinstance(result, CleaningResult) and not result.log.empty:
            display_frame(result.log, width="stretch", hide_index=True)
        else:
            st.caption("No se aplicó limpieza.")
    with st.expander("Ver perfil de columnas"):
        display_frame(cached_profile(df), width="stretch", hide_index=True, height=420)


# ---------------------------------------------------------------------------
# Sidebar y enrutamiento
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-logo">📊</div>'
            '<div><div class="brand-name">DataLimpia</div>'
            '<div class="brand-sub">Analytics</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">Navegación</div>', unsafe_allow_html=True)
        page = st.radio("Navegación", PAGES, key="nav", label_visibility="collapsed")

        st.markdown('<div class="side-label" style="margin-top:18px;">Dataset activo</div>', unsafe_allow_html=True)
        df = active_df()
        if df is None:
            st.markdown(
                '<div class="side-card"><div class="name">Sin datos</div>'
                '<div class="meta">Carga un archivo para comenzar</div></div>',
                unsafe_allow_html=True,
            )
        else:
            quality = quality_components(df)
            label, _ = quality_label(quality["puntaje"])
            color = "#34D399" if quality["puntaje"] >= 75 else "#FBBF24" if quality["puntaje"] >= 55 else "#F87171"
            state = "Limpio" if get_clean() is not None else "Original"
            rows_txt = fmt_number(len(df), 0)
            st.markdown(
                f'<div class="side-card"><div class="name">{html_lib.escape(truncate(st.session_state.get("raw_name", "dataset"), 30))}</div>'
                f'<div class="meta">{rows_txt} filas · {df.shape[1]} columnas</div>'
                f'<span class="side-pill" style="background:{color}22;color:{color};">{state} · {label} {quality["puntaje"]:.0f}/100</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("↺  Reiniciar aplicación", width="stretch"):
            reset_data()
        st.caption(f"{APP_NAME} · v{APP_VERSION}")
        st.caption(APP_TAGLINE)
    return page


def main() -> None:
    global _PLOT_COUNTER
    _PLOT_COUNTER = itertools.count()
    pending_page = st.session_state.pop("_pending_nav", None)
    if pending_page in PAGES:
        st.session_state["nav"] = pending_page
    apply_pending_reset()
    inject_css()
    page = render_sidebar()
    if page == PAGE_LOAD:
        render_load()
    elif page == PAGE_CLEAN:
        render_cleaning()
    elif page == PAGE_EXPLORE:
        render_explore()
    elif page == PAGE_DASH:
        render_dashboard()
    elif page == PAGE_REPORT:
        render_report()


if __name__ == "__main__":
    main()
