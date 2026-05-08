import sys


def leer_transacciones(lineas):

    productos = {}
    primera_linea = True

    for linea in lineas:
        linea = linea.strip()

        # Saltar encabezado
        if primera_linea:
            primera_linea = False
            continue

        # Saltar líneas vacías
        if not linea:
            continue

        # Parsear línea
        partes = linea.split(',')

        # Regla 5: ignorar líneas con menos de 4 columnas
        if len(partes) < 4:
            continue

        producto = partes[1]

        # Regla 5: ignorar líneas con cantidad o precio no numérico
        try:
            cantidad = int(partes[2])
            precio = float(partes[3])
        except ValueError:
            continue

        # Inicializar entrada si el producto es nuevo
        if producto not in productos:
            productos[producto] = {
                "unidades": 0,
                "ingreso": 0.0
            }

        # Acumular valores
        productos[producto]["unidades"] += cantidad
        productos[producto]["ingreso"] += cantidad * precio

    return productos


def calcular_promedios(productos):
    """
    Agrega el precio promedio a cada producto en el diccionario.
    
    Args:
        productos: dict con unidades e ingreso por producto
    
    Returns:
        el mismo dict, ahora con clave "promedio" agregada
    """
    for prod in productos:
        unidades = productos[prod]["unidades"]
        ingreso = productos[prod]["ingreso"]
        productos[prod]["promedio"] = ingreso / unidades if unidades > 0 else 0.0
    return productos


def ordenar_por_ingreso(productos):
    """
    Convierte el diccionario a lista ordenada por ingreso descendente.
    
    Args:
        productos: dict con datos por producto
    
    Returns:
        list of tuples: [(nombre, datos), ...] ordenado desc. por ingreso
    """
    return sorted(
        productos.items(),
        key=lambda x: x[1]["ingreso"],
        reverse=True
    )


def generar_csv(productos_ordenados):
    """
    Genera el CSV de salida como string.
    
    Args:
        productos_ordenados: lista de tuplas (nombre, datos)
    
    Returns:
        str: CSV con encabezado y una fila por producto
    """
    lineas = ["producto,unidades_vendidas,ingreso_total,precio_promedio"]

    for nombre, datos in productos_ordenados:
        unidades = datos["unidades"]
        ingreso = datos["ingreso"]
        promedio = datos["promedio"]
        lineas.append(f"{nombre},{unidades},{ingreso:.2f},{promedio:.2f}")

    return "\n".join(lineas)


def main():
    # Leer todas las líneas de stdin
    lineas = sys.stdin.readlines()

    # Procesar
    productos = leer_transacciones(lineas)
    productos = calcular_promedios(productos)
    productos_ordenados = ordenar_por_ingreso(productos)

    # Imprimir salida
    print(generar_csv(productos_ordenados))


if __name__ == "__main__":
    main()
