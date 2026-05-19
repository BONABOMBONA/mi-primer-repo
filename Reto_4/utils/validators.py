"""Validadores de datos para el sistema de inventario."""

import math


def validar_sku(sku):
    """True si el SKU no esta vacio."""
    if sku is None:
        return False
    return bool(str(sku).strip())


def validar_precio(precio):
    """
    True si el precio convierte a float finito y >= 0.
    Rechaza inf, -inf y NaN (aunque float() los acepte).
    """
    try:
        valor = float(precio)
    except (ValueError, TypeError):
        return False
    if not math.isfinite(valor):
        return False
    return valor >= 0


def validar_stock(stock):
    """True si el stock convierte a entero >= 0."""
    try:
        valor = int(stock)
    except (ValueError, TypeError):
        return False
    return valor >= 0


def validar_producto(sku, nombre, categoria, precio, stock, stock_minimo):
    """
    Valida todos los campos de un producto.

    Returns:
        tuple (es_valido: bool, mensaje_error: str | None)
    """
    if not validar_sku(sku):
        return False, "SKU vacio o invalido"

    if nombre is None or not str(nombre).strip():
        return False, "Nombre vacio"

    if categoria is None or not str(categoria).strip():
        return False, "Categoria vacia"

    if not validar_precio(precio):
        return False, f"Precio invalido: {precio}"

    if not validar_stock(stock):
        return False, f"Stock invalido: {stock}"

    if not validar_stock(stock_minimo):
        return False, f"Stock minimo invalido: {stock_minimo}"

    return True, None
