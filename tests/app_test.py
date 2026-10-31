"""Prueba de la interfaz con streamlit.testing (sin navegador).

Ejecutar con:  .venv\\Scripts\\python.exe tests\\app_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def show_exceptions(at: AppTest, step: str) -> None:
    if at.exception:
        print(f"[ERROR] {step}: {len(at.exception)} excepción(es)")
        for exc in at.exception:
            print("   ", exc.value)
        raise SystemExit(1)
    print(f"[OK] {step}")


def button(at: AppTest, text: str):
    matches = [b for b in at.button if text.lower() in b.label.lower()]
    return matches[0] if matches else None


def main() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    show_exceptions(at, "Carga inicial")
    assert any("demostración" in b.label for b in at.button), [b.label for b in at.button]

    demo = button(at, "demostración")
    demo.click().run()
    show_exceptions(at, "Carga del dataset de demostración")

    nav = [r for r in at.radio if r.key == "nav"][0]
    nav.set_value("🧹  Limpieza y calidad").run()
    show_exceptions(at, "Página de limpieza")
    assert button(at, "Aplicar limpieza") is not None

    button(at, "Aplicar limpieza").click().run()
    show_exceptions(at, "Aplicar limpieza")
    assert at.session_state["clean_result"] is not None
    print("    filas después de limpiar:", at.session_state["clean_result"].df.shape)

    nav = [r for r in at.radio if r.key == "nav"][0]
    nav.set_value("🔎  Explorar").run()
    show_exceptions(at, "Página de exploración")

    nav = [r for r in at.radio if r.key == "nav"][0]
    nav.set_value("📊  Dashboard").run()
    show_exceptions(at, "Dashboard")

    nav = [r for r in at.radio if r.key == "nav"][0]
    nav.set_value("📄  Informe").run()
    show_exceptions(at, "Página de informe")

    gen = button(at, "Generar informe")
    if gen is not None:
        gen.click().run()
        show_exceptions(at, "Generación de informe HTML")
        assert at.session_state["report_html"], "El informe HTML no se generó"

    print("\nInterfaz validada: carga, limpieza, exploración, dashboard e informe.")


if __name__ == "__main__":
    main()
