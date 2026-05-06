#!/usr/bin/env python3
"""
Validador de Códigos con Expresiones Regulares
Sistema de validación para códigos de producto, envío, empleado y factura.

Uso:
    python main.py
"""

import re
import sys
from datetime import date

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEPARTAMENTOS_VALIDOS = {"VEN", "ADM", "TEC", "LOG", "RHH"}
SERIES_VALIDAS = {"A", "B", "C", "D", "E"}

# ---------------------------------------------------------------------------
# PARTE 1: Funciones de validación individual
# ---------------------------------------------------------------------------

def validar_producto(codigo: str) -> dict:
    """
    Valida código de producto.
    Formato: ABC-1234-MX
        - Categoría : exactamente 3 letras mayúsculas
        - Número    : exactamente 4 dígitos
        - País      : exactamente 2 letras mayúsculas

    Returns:
        dict con claves: valido, categoria, numero, pais
    """
    resultado = {"valido": False, "categoria": None, "numero": None, "pais": None}

    patron = r"^([A-Z]{3})-(\d{4})-([A-Z]{2})$"
    m = re.match(patron, codigo)

    if m:
        resultado["valido"] = True
        resultado["categoria"] = m.group(1)
        resultado["numero"] = m.group(2)
        resultado["pais"] = m.group(3)

    return resultado


def validar_envio(codigo: str) -> dict:
    """
    Valida código de envío.
    Formato: ENV-YYYY-MM-DD-NNNNNN
        - Año  : 2020-2030
        - Mes  : 01-12
        - Día  : 01-31
        - Seq  : exactamente 6 dígitos

    Returns:
        dict con claves: valido, fecha, secuencial
    """
    resultado = {"valido": False, "fecha": None, "secuencial": None}

    patron = r"^ENV-(20[2-9]\d|2030)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])-(\d{6})$"
    m = re.match(patron, codigo)

    if m:
        anio, mes, dia = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if validar_fecha_real(anio, mes, dia):
            resultado["valido"] = True
            resultado["fecha"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            resultado["secuencial"] = m.group(4)

    return resultado


def validar_empleado(codigo: str) -> dict:
    """
    Valida código de empleado.
    Formato: EMP-XXX-NNNN
        - Departamento : VEN, ADM, TEC, LOG o RHH
        - Número       : 4 dígitos, no puede empezar con 0

    Returns:
        dict con claves: valido, departamento, numero
    """
    resultado = {"valido": False, "departamento": None, "numero": None}

    patron = r"^EMP-([A-Z]{3})-([1-9]\d{3})$"
    m = re.match(patron, codigo)

    if m and m.group(1) in DEPARTAMENTOS_VALIDOS:
        resultado["valido"] = True
        resultado["departamento"] = m.group(1)
        resultado["numero"] = m.group(2)

    return resultado


def validar_factura(codigo: str) -> dict:
    """
    Valida código de factura.
    Formato: FAC-S-NNNNNN
        - Serie  : A, B, C, D o E
        - Número : exactamente 6 dígitos

    Returns:
        dict con claves: valido, serie, numero
    """
    resultado = {"valido": False, "serie": None, "numero": None}

    patron = r"^FAC-([A-Z])-(\d{6})$"
    m = re.match(patron, codigo)

    if m and m.group(1) in SERIES_VALIDAS:
        resultado["valido"] = True
        resultado["serie"] = m.group(1)
        resultado["numero"] = m.group(2)

    return resultado


# ---------------------------------------------------------------------------
# PARTE 2: Validador universal
# ---------------------------------------------------------------------------

def validar_codigo(codigo: str) -> dict:
    """
    Detecta el tipo de código y lo valida automáticamente.

    Returns:
        dict con claves: codigo, tipo, valido, detalles
    """
    resultado = {
        "codigo": codigo,
        "tipo": "desconocido",
        "valido": False,
        "detalles": {},
    }

    if codigo.startswith("ENV-"):
        resultado["tipo"] = "envio"
        res = validar_envio(codigo)
        resultado["valido"] = res["valido"]
        if res["valido"]:
            resultado["detalles"] = {k: v for k, v in res.items() if k != "valido"}
        return resultado

    if codigo.startswith("EMP-"):
        resultado["tipo"] = "empleado"
        res = validar_empleado(codigo)
        resultado["valido"] = res["valido"]
        if res["valido"]:
            resultado["detalles"] = {k: v for k, v in res.items() if k != "valido"}
        return resultado

    if codigo.startswith("FAC-"):
        resultado["tipo"] = "factura"
        res = validar_factura(codigo)
        resultado["valido"] = res["valido"]
        if res["valido"]:
            resultado["detalles"] = {k: v for k, v in res.items() if k != "valido"}
        return resultado

    # Prefijo no reconocido: intentar como producto; si no encaja, desconocido
    resultado["tipo"] = "producto"
    res = validar_producto(codigo)
    if res["valido"]:
        resultado["valido"] = True
        resultado["detalles"] = {k: v for k, v in res.items() if k != "valido"}
    else:
        resultado["tipo"] = "desconocido"

    return resultado


# ---------------------------------------------------------------------------
# PARTE 3: Procesamiento por lotes
# ---------------------------------------------------------------------------

def procesar_lote(codigos: list) -> dict:
    """
    Procesa múltiples códigos y genera estadísticas.

    Returns:
        dict con total, validos, invalidos, por_tipo y detalle
    """
    resultado = {
        "total": 0,
        "validos": 0,
        "invalidos": 0,
        "por_tipo": {
            "producto":     {"total": 0, "validos": 0},
            "envio":        {"total": 0, "validos": 0},
            "empleado":     {"total": 0, "validos": 0},
            "factura":      {"total": 0, "validos": 0},
            "desconocido":  {"total": 0, "validos": 0},
        },
        "detalle": [],
    }

    for codigo in codigos:
        res = validar_codigo(codigo)
        tipo = res["tipo"]

        resultado["total"] += 1
        resultado["por_tipo"][tipo]["total"] += 1
        resultado["detalle"].append(res)

        if res["valido"]:
            resultado["validos"] += 1
            resultado["por_tipo"][tipo]["validos"] += 1
        else:
            resultado["invalidos"] += 1

    return resultado


# ---------------------------------------------------------------------------
# BONUS: Utilidades extras
# ---------------------------------------------------------------------------

def validar_fecha_real(anio: int, mes: int, dia: int) -> bool:
    """Valida que la combinación año/mes/día sea una fecha calendario real."""
    try:
        date(anio, mes, dia)
        return True
    except ValueError:
        return False


def sugerir_correccion(codigo: str) -> str:
    """
    Sugiere una corrección para códigos inválidos comunes.
    Actualmente normaliza a mayúsculas y prueba si resulta válido.
    """
    candidato = codigo.upper()
    if candidato != codigo:
        res = validar_codigo(candidato)
        if res["valido"]:
            return candidato
    return codigo


def exportar_resultados(reporte: dict, archivo: str) -> None:
    """Exporta el detalle de validación a un archivo CSV."""
    columnas = ["codigo", "tipo", "valido", "detalles"]
    with open(archivo, "w", encoding="utf-8") as f:
        f.write(",".join(columnas) + "\n")
        for item in reporte["detalle"]:
            detalles_str = "; ".join(
                f"{k}={v}" for k, v in item["detalles"].items()
            )
            f.write(
                f"{item['codigo']},{item['tipo']},{item['valido']},\"{detalles_str}\"\n"
            )


# ---------------------------------------------------------------------------
# Presentación de resultados
# ---------------------------------------------------------------------------

def mostrar_resultado(resultado: dict) -> None:
    """Muestra el resultado de validación de forma legible."""
    estado = "✓" if resultado["valido"] else "✗"
    print(f"{estado} {resultado['codigo']:<30} | Tipo: {resultado['tipo']:<12}")
    if resultado["valido"] and resultado["detalles"]:
        detalles = ", ".join(
            f"{k}: {v}" for k, v in resultado["detalles"].items() if v
        )
        print(f"   └── {detalles}")


def mostrar_reporte(reporte: dict) -> None:
    """Muestra el reporte de procesamiento por lotes."""
    total = reporte["total"]
    print("=" * 60)
    print("                 REPORTE DE VALIDACIÓN")
    print("=" * 60)
    print(f"\nTotal procesados: {total}")
    print(f"Válidos:   {reporte['validos']} ({reporte['validos']/total*100:.1f}%)")
    print(f"Inválidos: {reporte['invalidos']} ({reporte['invalidos']/total*100:.1f}%)")
    print("\nDesglose por tipo:")
    print("-" * 40)
    for tipo, stats in reporte["por_tipo"].items():
        if stats["total"] > 0:
            tasa = stats["validos"] / stats["total"] * 100
            print(
                f"  {tipo.capitalize():<12}: "
                f"{stats['validos']:>3}/{stats['total']:<3} ({tasa:.0f}% válidos)"
            )
    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Datos de prueba y punto de entrada
# ---------------------------------------------------------------------------

CODIGOS_PRUEBA = [
    # Productos
    "TEC-0001-MX",           # Válido
    "ALI-9999-US",           # Válido
    "ROB-1234-CA",           # Válido
    "tec-0001-MX",           # Inválido: minúsculas
    "TEC-001-MX",            # Inválido: solo 3 dígitos
    "TECH-0001-MX",          # Inválido: 4 letras en categoría
    # Envíos
    "ENV-2024-03-15-001234", # Válido
    "ENV-2025-12-01-999999", # Válido
    "ENV-2019-03-15-001234", # Inválido: año fuera de rango
    "ENV-2024-13-15-001234", # Inválido: mes 13
    "ENV-2024-03-32-001234", # Inválido: día 32
    # Empleados
    "EMP-VEN-1234",          # Válido
    "EMP-TEC-9999",          # Válido
    "EMP-ADM-1000",          # Válido
    "EMP-VEN-0123",          # Inválido: empieza con 0
    "EMP-XXX-1234",          # Inválido: departamento no válido
    "EMP-VEN-123",           # Inválido: solo 3 dígitos
    # Facturas
    "FAC-A-123456",          # Válido
    "FAC-E-000001",          # Válido
    "FAC-B-999999",          # Válido
    "FAC-F-123456",          # Inválido: serie F no existe
    "FAC-A-12345",           # Inválido: solo 5 dígitos
    "FAC-a-123456",          # Inválido: serie en minúscula
    # Desconocidos
    "XXX-1234",              # Desconocido
    "RANDOM-CODE",           # Desconocido
]


def main():
    print("PRUEBA DE FUNCIONES INDIVIDUALES")
    print("=" * 50)

    print("\n-- Productos --")
    print(validar_producto("TEC-0001-MX"))
    print(validar_producto("tec-0001-MX"))

    print("\n-- Envíos --")
    print(validar_envio("ENV-2024-03-15-001234"))
    print(validar_envio("ENV-2024-13-15-001234"))

    print("\n-- Empleados --")
    print(validar_empleado("EMP-VEN-1234"))
    print(validar_empleado("EMP-VEN-0123"))

    print("\n-- Facturas --")
    print(validar_factura("FAC-A-123456"))
    print(validar_factura("FAC-F-123456"))

    print("\n\nPRUEBA DE VALIDADOR UNIVERSAL")
    print("=" * 50)
    for codigo in CODIGOS_PRUEBA:
        mostrar_resultado(validar_codigo(codigo))

    print("\n\nPRUEBA DE PROCESAMIENTO POR LOTES")
    reporte = procesar_lote(CODIGOS_PRUEBA)
    mostrar_reporte(reporte)

    # BONUS: exportar a CSV
    exportar_resultados(reporte, "outputs/resultados_validacion.csv")
    print("Resultados exportados en: outputs/resultados_validacion.csv")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
