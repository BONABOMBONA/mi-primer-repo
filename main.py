"""
Clasificador de Temperaturas
Reto Semana 2 - Programacion para Ciencia de Datos
Instituto Politecnico Nacional | Semestre Febrero-Julio 2026

Lee un CSV desde stdin con columnas: ciudad,temperatura,unidad
Escribe un CSV a stdout con columnas: ciudad,temperatura_celsius,clasificacion
"""

import sys


def fahrenheit_a_celsius(f: float) -> float:
    """Convierte una temperatura de Fahrenheit a Celsius.
    
    Args:
        f: Temperatura en grados Fahrenheit.
    
    Returns:
        Temperatura equivalente en grados Celsius.
    """
    return (f - 32) * 5 / 9


def clasificar_temperatura(celsius: float) -> str:
    """Clasifica una temperatura en Celsius segun rangos predefinidos.
    
    Rangos:
        < 0       -> Congelante
        0  a 15   -> Frio
        16 a 25   -> Templado
        26 a 35   -> Calido
        > 35      -> Extremo
    
    Args:
        celsius: Temperatura en grados Celsius.
    
    Returns:
        Cadena con la clasificacion correspondiente.
    """
    if celsius < 0:
        return "Congelante"
    elif celsius <= 15:
        return "Frio"
    elif celsius <= 25:
        return "Templado"
    elif celsius <= 35:
        return "Calido"
    else:
        return "Extremo"


def procesar_linea(linea: str):
    """Parsea y procesa una linea del CSV de entrada.
    
    Args:
        linea: Cadena con el formato "ciudad,temperatura,unidad".
    
    Returns:
        Tupla (ciudad, celsius, clasificacion) si la linea es valida,
        None si contiene datos invalidos o esta incompleta.
    """
    partes = linea.strip().split(",")

    # Debe tener exactamente 3 columnas
    if len(partes) != 3:
        return None

    ciudad = partes[0].strip()
    temp_str = partes[1].strip()
    unidad = partes[2].strip().upper()

    # Validar unidad
    if unidad not in ("C", "F"):
        return None

    # Validar que la temperatura sea numerica
    try:
        temperatura = float(temp_str)
    except ValueError:
        return None

    # Convertir a Celsius si es necesario
    celsius = fahrenheit_a_celsius(temperatura) if unidad == "F" else temperatura

    clasificacion = clasificar_temperatura(celsius)
    return (ciudad, celsius, clasificacion)


def main():
    """Punto de entrada del programa.
    
    Lee datos desde stdin, descarta el encabezado de entrada,
    e imprime el CSV resultante en stdout.
    """
    primera_linea = True

    # Encabezado de salida
    print("ciudad,temperatura_celsius,clasificacion")

    for linea in sys.stdin:
        # Saltar encabezado de entrada
        if primera_linea:
            primera_linea = False
            continue

        # Saltar lineas vacias
        if not linea.strip():
            continue

        resultado = procesar_linea(linea)
        if resultado is None:
            continue  # Linea invalida -> ignorar silenciosamente

        ciudad, celsius, clasificacion = resultado
        print(f"{ciudad},{celsius:.1f},{clasificacion}")


if __name__ == "__main__":
    main()
