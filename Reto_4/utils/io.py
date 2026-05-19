"""Funciones de entrada y salida para el sistema de inventario."""


def leer_inventario(ruta_archivo):
    """
    Lee el archivo CSV de inventario y retorna una lista de diccionarios.

    - La primera linea se considera el encabezado.
    - Lineas vacias se ignoran.
    - Lineas con un numero de columnas distinto al del encabezado se ignoran
      (cubre los errores de columnas faltantes y columnas extra).

    Args:
        ruta_archivo: ruta al archivo CSV.

    Returns:
        list[dict]: lista de diccionarios (encabezado -> valor) por cada
        linea valida estructuralmente.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    productos_raw = []

    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        lineas = archivo.readlines()

    if not lineas:
        return productos_raw

    encabezados = [h.strip() for h in lineas[0].strip().split(",")]

    for linea in lineas[1:]:
        linea = linea.strip()
        if not linea:
            continue

        valores = [v.strip() for v in linea.split(",")]
        if len(valores) != len(encabezados):
            # Lineas con columnas faltantes o extra: ignorar.
            continue

        productos_raw.append(dict(zip(encabezados, valores)))

    return productos_raw


def escribir_reporte(productos, ruta_archivo):
    """
    Escribe el reporte de productos que necesitan reorden en formato CSV.

    Args:
        productos: lista de objetos Producto.
        ruta_archivo: ruta donde guardar el CSV.
    """
    encabezados = [
        "sku",
        "nombre",
        "categoria",
        "stock_actual",
        "stock_minimo",
        "unidades_faltantes",
        "valor_inventario",
    ]

    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        archivo.write(",".join(encabezados) + "\n")
        for p in productos:
            linea = (
                f"{p.sku},{p.nombre},{p.categoria},"
                f"{p.stock},{p.stock_minimo},"
                f"{p.unidades_faltantes()},{p.valor_inventario():.2f}"
            )
            archivo.write(linea + "\n")
