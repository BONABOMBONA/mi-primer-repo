#!/usr/bin/env python3
"""
Sistema de Inventario Modular.

Lee data/inventario.csv, identifica productos con stock < stock_minimo
y escribe el reporte en outputs/reporte_inventario.csv ordenado de
mayor a menor por unidades_faltantes.
"""

from models.producto import Producto
from utils.validators import validar_producto
from utils.io import leer_inventario, escribir_reporte

# Configuracion (rutas relativas: correr con `cd Reto_4 && python3 main.py`)
ARCHIVO_INVENTARIO = "data/inventario.csv"
ARCHIVO_REPORTE = "outputs/reporte_inventario.csv"


def crear_productos(datos_raw):
    """
    Convierte la lista de diccionarios crudos en objetos Producto.
    Ignora silenciosamente los registros invalidos.

    Args:
        datos_raw: Lista de dicts con los campos del CSV.

    Returns:
        list: Lista de objetos Producto validos.
    """
    productos = []

    for datos in datos_raw:
        es_valido, error = validar_producto(
            datos.get("sku"),
            datos.get("nombre"),
            datos.get("categoria"),
            datos.get("precio"),
            datos.get("stock"),
            datos.get("stock_minimo"),
        )

        if not es_valido:
            print(f"Advertencia: Ignorando registro invalido - {error}")
            continue

        productos.append(
            Producto(
                sku=datos["sku"],
                nombre=datos["nombre"],
                categoria=datos["categoria"],
                precio=float(datos["precio"]),
                stock=int(datos["stock"]),
                stock_minimo=int(datos["stock_minimo"]),
            )
        )

    return productos


def filtrar_necesitan_reorden(productos):
    """Filtra los productos cuyo stock esta por debajo del minimo."""
    return [p for p in productos if p.necesita_reorden()]


def ordenar_por_faltantes(productos):
    """Ordena por unidades_faltantes de forma descendente."""
    return sorted(productos, key=lambda p: p.unidades_faltantes(), reverse=True)


def main():
    print("=" * 50)
    print("SISTEMA DE INVENTARIO - Reporte de Reorden")
    print("=" * 50)

    # 1. Leer datos del CSV
    print(f"\nLeyendo inventario de: {ARCHIVO_INVENTARIO}")
    datos_raw = leer_inventario(ARCHIVO_INVENTARIO)
    print(f"Registros leidos: {len(datos_raw)}")

    # 2. Convertir a objetos Producto (descartando invalidos)
    productos = crear_productos(datos_raw)
    print(f"Productos validos: {len(productos)}")

    # 3. Filtrar los que necesitan reorden
    necesitan_reorden = filtrar_necesitan_reorden(productos)
    print(f"Productos que necesitan reorden: {len(necesitan_reorden)}")

    # 4. Ordenar por unidades faltantes descendente
    necesitan_reorden = ordenar_por_faltantes(necesitan_reorden)

    # 5. Mostrar resumen en consola
    print("\n" + "-" * 50)
    print("PRODUCTOS QUE NECESITAN REORDEN:")
    print("-" * 50)
    for p in necesitan_reorden:
        print(p)

    # 6. Escribir reporte CSV
    escribir_reporte(necesitan_reorden, ARCHIVO_REPORTE)
    print(f"\nReporte guardado en: {ARCHIVO_REPORTE}")

    print("\n" + "=" * 50)
    print("Proceso completado exitosamente")
    print("=" * 50)


if __name__ == "__main__":
    main()
