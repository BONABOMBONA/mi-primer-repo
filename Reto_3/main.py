"""
Reto Semana 3: Analizador de Ventas
Programacion para Ciencia de Datos - IPN 2026
"""

import sys
import math


def parsear_linea(linea):
    """Devuelve (producto, cantidad, precio) si la linea es valida, o None."""
    partes = linea.split(",")

    # Debe tener exactamente 4 columnas
    if len(partes) != 4:
        return None

    fecha, producto, cant_str, prec_str = (p.strip() for p in partes)

    # Producto no puede estar vacio
    if not producto:
        return None

    # Cantidad: entero positivo
    try:
        cantidad = int(cant_str)
    except ValueError:
        return None
    if cantidad <= 0:
        return None

    # Precio: flotante finito y positivo (rechaza inf, -inf, NaN)
    try:
        precio = float(prec_str)
    except ValueError:
        return None
    if not math.isfinite(precio) or precio <= 0:
        return None

    return producto, cantidad, precio


def leer_transacciones(lineas):
    """Agrupa transacciones por producto (unidades e ingreso acumulados)."""
    productos = {}
    primera = True

    for linea in lineas:
        linea = linea.strip()

        # Saltar encabezado
        if primera:
            primera = False
            continue

        # Saltar lineas vacias
        if not linea:
            continue

        resultado = parsear_linea(linea)
        if resultado is None:
            continue

        producto, cantidad, precio = resultado

        if producto not in productos:
            productos[producto] = {"unidades": 0, "ingreso": 0.0}

        productos[producto]["unidades"] += cantidad
        productos[producto]["ingreso"] += cantidad * precio

    return productos


def calcular_promedios(productos):
    """Agrega la clave 'promedio' a cada producto."""
    for prod in productos:
        unidades = productos[prod]["unidades"]
        ingreso = productos[prod]["ingreso"]
        productos[prod]["promedio"] = ingreso / unidades if unidades > 0 else 0.0
    return productos


def ordenar_por_ingreso(productos):
    """Lista de tuplas (nombre, datos) ordenada por ingreso desc."""
    return sorted(
        productos.items(),
        key=lambda x: x[1]["ingreso"],
        reverse=True,
    )


def generar_csv(productos_ordenados):
    """Genera el CSV de salida como string."""
    lineas = ["producto,unidades_vendidas,ingreso_total,precio_promedio"]
    for nombre, datos in productos_ordenados:
        lineas.append(
            f"{nombre},{datos['unidades']},{datos['ingreso']:.2f},{datos['promedio']:.2f}"
        )
    return "\n".join(lineas)


def main():
    lineas = sys.stdin.readlines()

    productos = leer_transacciones(lineas)
    productos = calcular_promedios(productos)
    productos_ordenados = ordenar_por_ingreso(productos)

    print(generar_csv(productos_ordenados))


if __name__ == "__main__":
    main()
