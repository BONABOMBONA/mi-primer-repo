"""Clase Producto del sistema de inventario."""


class Producto:
    """
    Representa un producto en el inventario.

    Attributes:
        sku (str): Identificador unico del producto.
        nombre (str): Nombre del producto.
        categoria (str): Categoria del producto.
        precio (float): Precio unitario (>= 0).
        stock (int): Cantidad actual en inventario (>= 0).
        stock_minimo (int): Nivel minimo antes de reordenar (>= 0).
    """

    def __init__(self, sku, nombre, categoria, precio, stock, stock_minimo):
        self.sku = sku
        self.nombre = nombre
        self.categoria = categoria
        self.precio = precio
        self.stock = stock
        self.stock_minimo = stock_minimo

    def necesita_reorden(self):
        """True si stock < stock_minimo."""
        return self.stock < self.stock_minimo

    def unidades_faltantes(self):
        """Unidades faltantes para alcanzar el stock minimo (0 si no aplica)."""
        if self.necesita_reorden():
            return self.stock_minimo - self.stock
        return 0

    def valor_inventario(self):
        """Valor monetario del inventario actual: precio * stock."""
        return self.precio * self.stock

    def __str__(self):
        estado = "[REORDEN]" if self.necesita_reorden() else "[OK]"
        return f"{estado} {self.sku}: {self.nombre} - Stock: {self.stock}/{self.stock_minimo}"

    def __repr__(self):
        return (
            f"Producto('{self.sku}', '{self.nombre}', '{self.categoria}', "
            f"{self.precio}, {self.stock}, {self.stock_minimo})"
        )
