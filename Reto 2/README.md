# Clasificador de Temperaturas

**Reto Semana 2 — Programación para Ciencia de Datos**  
Instituto Politécnico Nacional | Semestre Febrero-Julio 2026

---

## Descripción

Programa de línea de comandos que unifica y clasifica temperaturas de ciudades del mundo.  
Lee un CSV con datos en Celsius o Fahrenheit desde **stdin** y produce un CSV estandarizado en **stdout**.

```
╔════════════════════════════════════════════════════════════════╗
║               CLASIFICADOR DE TEMPERATURAS                     ║
╠════════════════════════════════════════════════════════════════╣
║  Datos mixtos (C y F)  →  Convierte + Clasifica  →  Solo °C   ║
║                                                                ║
║  CDMX,22,C          ──────────────────────>  CDMX,22.0,Templado  ║
║  NYC,50,F                                   NYC,10.0,Frio        ║
║  Moscow,-10,C                               Moscow,-10.0,Congelante ║
╚════════════════════════════════════════════════════════════════╝
```

---

## Estructura del Repositorio

```
reto-semana-02/
├── README.md                   # Este archivo
├── main.py                     # Solución principal
├── .gitignore                  # Archivos a ignorar
└── tests/
    ├── entrada1.txt            # Datos de prueba
    └── salida_esperada1.txt    # Salida esperada
```

---

## Formato de Entrada

CSV con tres columnas, leído desde **stdin**:

| Columna | Tipo | Descripción | Ejemplos |
|---------|------|-------------|----------|
| `ciudad` | texto | Nombre de la ciudad | `CDMX`, `Nueva York` |
| `temperatura` | número | Valor numérico | `22`, `-5`, `98.6` |
| `unidad` | texto | `C` o `F` (mayúsculas o minúsculas) | `C`, `f` |

**Ejemplo:**
```
ciudad,temperatura,unidad
CDMX,22,C
Nueva York,50,F
Moscu,-10,C
```

---

## Formato de Salida

CSV con tres columnas, escrito a **stdout**:

| Columna | Descripción |
|---------|-------------|
| `ciudad` | Nombre original |
| `temperatura_celsius` | Temperatura convertida a °C (1 decimal) |
| `clasificacion` | Categoría según tabla de rangos |

### Tabla de Clasificación

| Rango (°C) | Clasificación |
|-----------|---------------|
| < 0 | Congelante |
| 0 a 15 | Frio |
| 16 a 25 | Templado |
| 26 a 35 | Calido |
| > 35 | Extremo |

---

## Cómo Ejecutar

### Requisitos

- Python 3.6 o superior (usa f-strings)
- Sin dependencias externas

### Ejecución en Linux/Mac (Ubuntu)

```bash
# Con archivo de prueba
python main.py < tests/entrada1.txt

# Verificar contra salida esperada
python main.py < tests/entrada1.txt | diff - tests/salida_esperada1.txt

# Entrada manual (terminar con Ctrl+D)
python main.py
```

### Ejecución en Windows

```powershell
# PowerShell
Get-Content tests\entrada1.txt | python main.py

# CMD
type tests\entrada1.txt | python main.py
```

---

## Ejemplo Completo

**Entrada (`tests/entrada1.txt`):**
```
ciudad,temperatura,unidad
CDMX,22,C
Nueva York,50,F
Moscu,-10,C
Miami,95,F
Cancun,30,C
Chicago,14,F
Phoenix,104,F
Error,abc,C
```

**Salida:**
```
ciudad,temperatura_celsius,clasificacion
CDMX,22.0,Templado
Nueva York,10.0,Frio
Moscu,-10.0,Congelante
Miami,35.0,Calido
Cancun,30.0,Calido
Chicago,-10.0,Congelante
Phoenix,40.0,Extremo
```
> La línea con `Error,abc,C` fue ignorada (temperatura no numérica).

---

## Manejo de Datos Inválidos

Las siguientes líneas se ignoran silenciosamente:

- Temperatura no numérica (`abc`, `--`, vacío)
- Unidad diferente a `C` o `F` (ej. `X`, `K`)
- Menos de 3 columnas
- Líneas vacías

---

## Fórmula de Conversión

```
        (°F − 32) × 5
°C  =  ───────────────
              9
```

---

Dana Paola Soria Lòpez  
Programación para Ciencia de Datos — IPN 2026
