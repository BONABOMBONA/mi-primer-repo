"""Validadores de datos."""

def validar_sku(sku):
    return bool(sku and str(sku).strip())

def validar_precio(precio):
    try:
        return float(precio) >= 0
    except (ValueError, TypeError):
        return False

def validar_stock(stock):
    try:
        return int(stock) >= 0
    except (ValueError, TypeError):
        return False

def validar_producto(sku, nombre, categoria, precio, stock, stock_minimo):
    if not validar_sku(sku):
        return False, "SKU invalido"
    if not nombre or not str(nombre).strip():
        return False, "Nombre vacio"
    if not validar_precio(precio):
        return False, f"Precio invalido: {precio}"
    if not validar_stock(stock):
        return False, f"Stock invalido: {stock}"
    if not validar_stock(stock_minimo):
        return False, f"Stock minimo invalido: {stock_minimo}"
    return True, None
