"""Capturas de pantalla de la aplicación para revisión visual.

Ejecutar con:  .venv\\Scripts\\python.exe tests\\visual_test.py
Requiere: playwright install chromium
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)
URL = "http://localhost:8501"


def wait_app(page) -> None:
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=60000)
    page.wait_for_timeout(1800)


def click_nav(page, text: str) -> None:
    page.locator('[data-testid="stSidebar"] label').filter(has_text=text).first.click()
    page.wait_for_timeout(2200)


def main() -> None:
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda exc: errors.append(f"PAGEERROR: {exc}"))
        page.on(
            "console",
            lambda msg: errors.append(f"CONSOLE {msg.type}: {msg.text}")
            if msg.type == "error"
            else None,
        )

        page.goto(URL, wait_until="domcontentloaded")
        wait_app(page)
        page.screenshot(path=str(OUT / "01_inicio.png"), full_page=True)
        print("captura 01_inicio")

        page.get_by_role("button", name="Probar con un dataset de demostración").click()
        page.wait_for_selector("text=Dataset cargado", timeout=60000)
        page.wait_for_timeout(1800)
        page.screenshot(path=str(OUT / "02_cargado.png"), full_page=True)
        print("captura 02_cargado")

        click_nav(page, "Limpieza y calidad")
        page.wait_for_selector("text=Diagnóstico", timeout=30000)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "03_limpieza.png"), full_page=True)
        print("captura 03_limpieza")

        page.get_by_role("button", name="Aplicar limpieza").click()
        page.wait_for_selector("text=Resultado de la limpieza", timeout=90000)
        page.wait_for_timeout(1800)
        page.screenshot(path=str(OUT / "04_resultado.png"), full_page=True)
        print("captura 04_resultado")

        click_nav(page, "Dashboard")
        page.wait_for_selector("text=Métrica principal", timeout=30000)
        page.wait_for_timeout(2600)
        page.screenshot(path=str(OUT / "05_dashboard.png"), full_page=True)
        print("captura 05_dashboard")

        click_nav(page, "Informe")
        page.wait_for_selector("text=Resumen ejecutivo", timeout=30000)
        page.wait_for_timeout(2600)
        page.screenshot(path=str(OUT / "06_informe.png"), full_page=True)
        print("captura 06_informe")

        browser.close()

    critical = [e for e in errors if "favicon" not in e.lower()]
    if critical:
        print("\nErrores de consola:")
        for e in critical[:20]:
            print(" ", e)
    else:
        print("\nSin errores de consola.")


if __name__ == "__main__":
    main()
