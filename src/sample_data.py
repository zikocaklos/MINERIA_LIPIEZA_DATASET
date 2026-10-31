"""Dataset de demostración con problemas típicos de datos reales."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_sample_dataset(rows: int = 1800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regions = ["Norte", "Centro", "Sur", "Occidente", "Oriente"]
    region_p = [0.30, 0.25, 0.20, 0.15, 0.10]
    channels = ["Online", "Tienda", "Distribuidor", "Mayorista"]
    channel_p = [0.36, 0.30, 0.20, 0.14]
    products = {
        "Laptop Pro 14": ("Tecnología", 1200, 0.18),
        "Monitor 27\"": ("Tecnología", 320, 0.22),
        "Teclado mecánico": ("Tecnología", 85, 0.30),
        "Silla ergonómica": ("Mobiliario", 240, 0.26),
        "Escritorio ajustable": ("Mobiliario", 480, 0.24),
        "Lámpara LED": ("Mobiliario", 45, 0.35),
        "Audífonos BT": ("Accesorios", 110, 0.32),
        "Mouse inalámbrico": ("Accesorios", 42, 0.38),
    }
    product_names = list(products)
    product_p = [0.10, 0.14, 0.12, 0.14, 0.10, 0.12, 0.16, 0.12]
    sellers = ["Ana Ruiz", "Carlos Pérez", "María Gómez", "Julián Díaz", "Sofía Torres",
               "Diego Ramírez", "Laura Castro", "Andrés Molina"]
    payments = ["Tarjeta", "Transferencia", "Efectivo", "Crédito"]

    start = pd.Timestamp("2024-01-01")
    days = (pd.Timestamp("2025-12-31") - start).days
    offsets = rng.integers(0, days + 1, size=rows)
    dates = pd.Series(start + pd.to_timedelta(offsets, unit="D"))

    region = rng.choice(regions, size=rows, p=region_p)
    channel = rng.choice(channels, size=rows, p=channel_p)
    product = rng.choice(product_names, size=rows, p=product_p)

    # Estacionalidad: más ventas en noviembre y diciembre.
    month = dates.dt.month
    seasonal = np.where(month == 12, 1.55, np.where(month == 11, 1.35, 1.0))
    base_units = rng.integers(1, 16, size=rows).astype(float)
    channel_boost = np.where(channel == "Distribuidor", 2.4, np.where(channel == "Mayorista", 1.9, 1.0))
    units = np.maximum(1, np.round(base_units * seasonal * channel_boost)).astype(int)

    price = np.array([products[p][1] for p in product], dtype=float)
    cost_share = np.array([products[p][2] for p in product], dtype=float)
    price = price * rng.normal(1.0, 0.05, size=rows)
    price = np.round(np.maximum(price, 5), 2)

    discount = np.round(rng.choice([0, 0, 0, 0.05, 0.10, 0.15], size=rows), 2)
    ingreso = np.round(units * price * (1 - discount), 2)
    costo_total = np.round(ingreso * (1 - cost_share) * rng.normal(1.0, 0.04, size=rows), 2)
    margen = np.round(ingreso - costo_total, 2)
    shipping = np.round(rng.gamma(2.0, 12.0, size=rows) + units * 1.5, 2)

    satisfaction = np.clip(
        np.round(
            3.9
            + np.where(channel == "Online", 0.55, 0.0)
            - np.where(channel == "Mayorista", 0.35, 0.0)
            + rng.normal(0, 0.7, size=rows)
        ),
        1,
        5,
    ).astype(int)

    category = [products[p][0] for p in product]
    seller = rng.choice(sellers, size=rows)
    payment = rng.choice(payments, size=rows, p=[0.45, 0.3, 0.15, 0.10])

    # --- Suciedad intencional ------------------------------------------------
    def dirty_text(values: np.ndarray, share: float = 0.12) -> np.ndarray:
        out = values.astype(object).copy()
        idx = rng.choice(rows, size=int(rows * share), replace=False)
        for i in idx:
            variant = rng.integers(0, 2)
            value = str(out[i])
            out[i] = f"  {value} " if variant == 0 else f"{value} "
        return out

    region = dirty_text(region)
    channel = dirty_text(channel, 0.06)
    product = dirty_text(product, 0.08)

    # Nulos y nulos encubiertos.
    channel_idx = rng.choice(rows, size=int(rows * 0.06), replace=False)
    channel[channel_idx] = "N/A"
    channel[rng.choice(rows, size=int(rows * 0.02), replace=False)] = "-"

    units_series = np.array([str(int(u)) for u in units], dtype=object)
    units_series[rng.choice(rows, size=int(rows * 0.03), replace=False)] = "N/D"

    # Precios con formato de moneda latinoamericano.
    price_series = np.array([f"{p:.2f}" for p in price], dtype=object)
    money_idx = rng.choice(rows, size=int(rows * 0.12), replace=False)
    price_series[money_idx] = [
        "$" + f"{price[i]:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        for i in money_idx
    ]
    price_series[rng.choice(rows, size=int(rows * 0.03), replace=False)] = "N/A"

    ingreso_series = np.round(ingreso.astype(float), 2)
    ingreso_series[rng.choice(rows, size=int(rows * 0.04), replace=False)] = np.nan

    satisfaction_series = satisfaction.astype(object)
    satisfaction_series[rng.choice(rows, size=int(rows * 0.06), replace=False)] = np.nan
    satisfaction_series[rng.choice(rows, size=int(rows * 0.02), replace=False)] = "sin dato"

    payment_series = payment.astype(object)
    payment_series[rng.choice(rows, size=int(rows * 0.05), replace=False)] = np.nan

    shipping_series = shipping.copy()
    outliers = rng.choice(rows, size=max(3, int(rows * 0.01)), replace=False)
    shipping_series[outliers] = shipping_series[outliers] * rng.uniform(8, 15, size=len(outliers))
    shipping_series = np.round(shipping_series, 2)

    # Fechas mixtas: ISO y dd/mm/aaaa.
    date_text = dates.dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    latin_idx = rng.choice(rows, size=int(rows * 0.10), replace=False)
    date_text[latin_idx] = dates.iloc[latin_idx].dt.strftime("%d/%m/%Y").to_numpy(dtype=object)
    date_text[rng.choice(rows, size=max(2, int(rows * 0.01)), replace=False)] = "sin fecha"

    comments = np.full(rows, np.nan, dtype=object)
    comments[rng.choice(rows, size=max(5, int(rows * 0.04)), replace=False)] = rng.choice(
        ["Entrega rápida", "Cliente frecuente", "Solicitó factura", "Retraso menor"], size=max(5, int(rows * 0.04))
    )

    df = pd.DataFrame(
        {
            "ID Venta": [f"V-{i:05d}" for i in range(1, rows + 1)],
            "Fecha": date_text,
            "Region": region,
            "Canal": channel,
            "Producto": product,
            "Categoria": category,
            "Vendedor": seller,
            "Cliente ID": [f"C-{i:04d}" for i in rng.integers(1, 700, size=rows)],
            "Unidades": units_series,
            "Precio Unitario": price_series,
            "Descuento": discount,
            "Ingreso": ingreso_series,
            "Costo Total": costo_total,
            "Margen": margen,
            "Costo Envio": shipping_series,
            "Satisfaccion": satisfaction_series,
            "Metodo Pago": payment_series,
            "Pais": ["Colombia"] * rows,
            "Comentarios": comments,
        }
    )

    # Filas duplicadas reales (mismo contenido).
    duplicate_idx = rng.choice(rows, size=int(rows * 0.02), replace=False)
    df = pd.concat([df, df.iloc[duplicate_idx]], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def save_sample_csv(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_sample_dataset().to_csv(path, index=False, encoding="utf-8-sig")
    return path


if __name__ == "__main__":
    target = save_sample_csv(Path(__file__).resolve().parents[1] / "sample_data" / "ventas_demo.csv")
    print(f"Dataset de ejemplo generado en: {target}")
