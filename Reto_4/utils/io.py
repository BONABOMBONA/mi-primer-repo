"""Funciones de lectura y escritura."""

def leer_inventario(ruta_archivo):
    productos_raw = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
    if not lineas:
        return productos_raw
    encabezados = lineas[0].strip().split(',')
    for linea in lineas[1:]:
        linea = linea.strip()
        if not linea:
            continue
        valores = linea.split(',')
        if len(valores) == len(encabezados):
            productos_raw.append(dict(zip(encabezados, valores)))
    return productos_raw

def escribir_reporte(productos, ruta_archivo):
    encabezados = ["sku","nombre","categoria","stock_actual","stock_minimo","unidades_faltantes","valor_inventario"]
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write(','.join(encabezados) + '\n')
        for p in productos:
            f.write(f"{p.sku},{p.nombre},{p.categoria},{p.stock},{p.stock_minimo},{p.unidades_faltantes()},{p.valor_inventario():.2f}\n")
