import sys

def limpiar_valor(valor):

    # Ignorar Espacios
    valor = valor.strip()
    
    # Limpiar Caracteres Inválidos
    caracteres_validos = '0123456789.-'
    limpio = ''
    for char in valor:
        if char in caracteres_validos:
            limpio += char
            
    # Si después de limpiar no queda nada es 0
    if not limpio:
        return 0
        
    try:
        return int(float(limpio))
    except ValueError:
        # Captura errores
        return 0

def procesar_linea(linea):

    # Líneas Vacías
    linea = linea.strip()
    if not linea:
        return 0
        
    # Múltiples Elementos 
    valores = linea.split(',')
    suma = 0
    
    for v in valores:
        suma += limpiar_valor(v)
        
    return suma

def main():
    for linea in sys.stdin:
        resultado = procesar_linea(linea)
        print(resultado)

if __name__ == "__main__":
    main()
