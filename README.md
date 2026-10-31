# 📊 DataLimpia Analytics

Aplicación de escritorio/web hecha con **Python + Streamlit** para convertir un dataset
desordenado en información útil para decidir: limpia los datos, calcula indicadores, genera
gráficas automáticas y exporta un informe listo para compartir.

## ¿Qué hace?

1. **Carga** archivos CSV, Excel (`.xlsx`/`.xls`), JSON o Parquet (hasta 300 MB).
2. **Diagnostica** el dataset: nulos, duplicados, nulos encubiertos (`N/A`, `-`, `sin dato`),
   espacios, mayúsculas inconsistentes, números/fechas guardados como texto, columnas
   constantes, alta cardinalidad y valores atípicos.
3. **Limpia** con reglas configurables y una **bitácora de cada cambio**:
   - Normalización de nombres de columnas.
   - Limpieza de texto y conversión de nulos encubiertos.
   - Conversión automática de tipos (números con `$`, `%`, separadores `1.234,56`; fechas).
   - Eliminación de duplicados, filas/columnas vacías y columnas constantes.
   - Eliminación configurable de columnas binarias (0/1) con al menos 90% de ceros
     (umbral ajustable entre 90% y 99%) y detección por nombre de columnas de credenciales
     (contraseñas, tokens, claves y firmas); estas se ocultan en las vistas del dataset
     original y se eliminan por defecto al limpiar.
   - Imputación de nulos (mediana, media, moda, constante, forward fill, eliminar).
   - Tratamiento de atípicos por IQR o Z-score (recortar, eliminar o marcar).
4. **Explora** con filtros por texto, categoría, rango de fechas y rango numérico.
5. **Dashboard automático** con KPIs, distribuciones, correlaciones, Pareto (80/20),
   evolución temporal, comparación por categorías y un constructor de gráficas.
6. **Genera hallazgos** en lenguaje natural con acciones sugeridas (calidad, concentración,
   tendencia, correlaciones, atípicos y más).
7. **Exporta** CSV limpio, Excel con formato (datos + resumen + perfil + bitácora) e
   **informe HTML autocontenido** que se puede imprimir a PDF.

Todo el procesamiento ocurre en el equipo local; no se envían datos a servicios externos.

## Requisitos

- Python 3.11 o superior (probado con Python 3.14).
- Windows, macOS o Linux.

## Instalación y ejecución

```powershell
cd data-analytics-studio

# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar el entorno (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
python -m pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

En Windows también puedes hacer doble clic en **`iniciar.bat`**: crea el entorno, instala
dependencias si hace falta y abre la aplicación en el navegador.

> Los entornos virtuales de Python no se pueden copiar entre computadoras. Si moviste el
> proyecto, usa `iniciar.bat`: detecta un entorno inválido, lo conserva como `.venv-anterior`
> y crea uno nuevo con el Python instalado en este equipo.

### Sin interfaz gráfica (procesamiento por lotes)

```powershell
.\.venv\Scripts\python.exe procesar.py "C:\ruta\ventas.xlsx" --salida "C:\ruta\resultados"
```

Genera `*_limpio.csv`, `*_informe.xlsx` e `*_informe.html` con la configuración de limpieza
recomendada.

## Uso recomendado

1. **Cargar datos** → sube el archivo o pulsa «Probar con un dataset de demostración».
2. **Limpieza y calidad** → revisa el diagnóstico y aplica la limpieza. Los valores por
   defecto resuelven los problemas más comunes; ajusta las reglas si lo necesitas.
3. **Explorar** → filtra y valida casos puntuales.
4. **Dashboard** → elige la métrica, la dimensión y la fecha; revisa los hallazgos.
5. **Informe** → descarga CSV/Excel o genera el informe HTML.

> Consejo: la primera vez, usa el dataset de demostración (`sample_data/ventas_demo.csv`)
> para ver el flujo completo con 1.800 ventas y problemas de calidad realistas.

## Estructura del proyecto

```
data-analytics-studio/
├─ app.py                     # Interfaz Streamlit y navegación
├─ procesar.py                # Procesamiento por lotes (CLI)
├─ iniciar.bat                # Lanzador para Windows
├─ requirements.txt
├─ .streamlit/config.toml     # Tema y límites de subida
├─ src/
│  ├─ config.py               # Constantes, paleta y tokens de nulos
│  ├─ io_utils.py             # Lectura de CSV/Excel/JSON/Parquet
│  ├─ cleaning.py             # Motor de limpieza y bitácora
│  ├─ profiling.py            # Calidad, diagnóstico y perfilado
│  ├─ charts.py               # Gráficas Plotly
│  ├─ insights.py             # Hallazgos y recomendaciones
│  ├─ report.py               # Exportación CSV/Excel/HTML
│  ├─ sample_data.py          # Generador del dataset de demostración
│  └─ utils.py                # Utilidades de tipos y formato
├─ sample_data/
│  └─ ventas_demo.csv         # Dataset de ejemplo
└─ tests/
   ├─ smoke_test.py           # Pruebas del motor, gráficas y exportaciones
   └─ app_test.py             # Pruebas de la interfaz con Streamlit AppTest
```

## Pruebas

```powershell
.\.venv\Scripts\python.exe tests\smoke_test.py
.\.venv\Scripts\python.exe tests\app_test.py
```

## Notas y límites

- Las correlaciones **no implican causalidad**; valida los hallazgos con el negocio.
- La imputación y el recorte de atípicos modifican los datos: conserva siempre el archivo
  original y revisa la bitácora.
- Para archivos muy grandes (más de ~1 millón de filas) algunas gráficas y el perfilado
  pueden tardar; conviene filtrar o agregar antes de graficar.
- El informe HTML incluye Plotly incrustado (~5 MB) para que funcione sin internet.
- Formatos de fecha ambiguos (`01/05/2024`) se interpretan como día/mes/año por defecto;
  puedes cambiarlo en la configuración de tipos.

## Tecnologías

Python · pandas · NumPy · Streamlit · Plotly · XlsxWriter · PyArrow.
