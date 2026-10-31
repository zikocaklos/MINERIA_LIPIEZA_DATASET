"""Configuración compartida de la aplicación."""

APP_NAME = "DataLimpia Analytics"
APP_TAGLINE = "Limpieza, exploración y visualización para decidir mejor"
APP_VERSION = "1.0.0"

# Paleta corporativa usada en todas las gráficas.
PALETTE = [
    "#2563EB",  # azul
    "#14B8A6",  # teal
    "#F59E0B",  # ámbar
    "#8B5CF6",  # violeta
    "#EF4444",  # rojo
    "#06B6D4",  # cian
    "#84CC16",  # lima
    "#F97316",  # naranja
    "#EC4899",  # rosa
    "#64748B",  # gris
]

CHART_TEMPLATE = "plotly_white"

# Valores que suelen representar "sin dato" en archivos reales.
PLACEHOLDER_NULLS = {
    "",
    "na",
    "n/a",
    "n.a.",
    "n.a",
    "nan",
    "null",
    "none",
    "nil",
    "-",
    "--",
    "---",
    "?",
    "s/d",
    "sd",
    "sin dato",
    "sin datos",
    "sin informacion",
    "sin información",
    "no aplica",
    "no informado",
    "no disponible",
    "#n/a",
    "#¡valor!",
    "#valor!",
    "undefined",
}

# Palabras que sugieren que una columna es un identificador y no debe
# convertirse automáticamente a número (perdería ceros a la izquierda).
ID_HINTS = (
    "id",
    "ids",
    "codigo",
    "código",
    "cod",
    "sku",
    "zip",
    "postal",
    "telefono",
    "teléfono",
    "tel",
    "celular",
    "dni",
    "ruc",
    "nit",
    "cuenta",
    "referencia",
    "ref",
    "factura",
    "documento",
)

DATE_HINTS = (
    "fecha",
    "date",
    "dia",
    "día",
    "periodo",
    "período",
    "mes",
    "anio",
    "año",
    "year",
    "time",
    "timestamp",
    "created",
    "creado",
)

QUALITY_WEIGHTS = {
    "completitud": 0.35,
    "unicidad": 0.15,
    "columnas_utiles": 0.10,
    "consistencia": 0.40,
}
