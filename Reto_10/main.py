import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# =====================================================================
# PARTE 1: Análisis Estadístico Básico
# =====================================================================

def estadisticas_basicas(precios: pd.Series) -> Dict:
    """
    Calcula estadísticas descriptivas de los precios.
    """
    resultado = {
        "precio_actual": float(precios.iloc[-1]),
        "precio_minimo": float(precios.min()),
        "precio_maximo": float(precios.max()),
        "precio_promedio": float(precios.mean()),
        "precio_mediana": float(precios.median()),
        "desviacion_std": float(precios.std()),
        "rango": float(precios.max() - precios.min()),
        "dias_analizados": int(precios.count())
    }
    return resultado

def calcular_rendimientos(precios: pd.Series) -> pd.Series:
    """
    Calcula el rendimiento diario porcentual.
    """
    # Usamos .pct_change() y multiplicamos por 100 para tenerlo en porcentaje (%)
    return precios.pct_change() * 100

def analisis_rendimientos(rendimientos: pd.Series) -> Dict:
    """
    Analiza los rendimientos calculados sin tomar en cuenta el primer NaN.
    """
    # Eliminamos el primer valor que por defecto es NaN debido a pct_change()
    rend_limpios = rendimientos.dropna()
    
    if rend_limpios.empty:
        return {}

    # Encontrar el mejor y peor día (fecha y valor)
    fecha_mejor = rend_limpios.idxmax()
    mejor_dia = (fecha_mejor.strftime('%Y-%m-%d'), float(rend_limpios.loc[fecha_mejor]))
    
    fecha_peor = rend_limpios.idxmin()
    peor_dia = (fecha_peor.strftime('%Y-%m-%d'), float(rend_limpios.loc[fecha_peor]))

    resultado = {
        "rendimiento_total": float(rend_limpios.sum()),
        "rendimiento_promedio": float(rend_limpios.mean()),
        "mejor_dia": mejor_dia,
        "peor_dia": peor_dia,
        "dias_positivos": int((rend_limpios > 0).sum()),
        "dias_negativos": int((rend_limpios < 0).sum()),
        "volatilidad": float(rend_limpios.std())
    }
    return resultado

# =====================================================================
# PARTE 2: Indicadores Técnicos
# =====================================================================

def media_movil(precios: pd.Series, ventana: int) -> pd.Series:
    """
    Calcula la media móvil simple (SMA).
    """
    return precios.rolling(window=ventana).mean()

def bandas_bollinger(precios: pd.Series, ventana: int = 20, num_std: int = 2) -> Dict:
    """
    Calcula las Bandas de Bollinger.
    """
    sma = precios.rolling(window=ventana).mean()
    std = precios.rolling(window=ventana).std()
    
    resultado = {
        "banda_superior": sma + (num_std * std),
        "banda_media": sma,
        "banda_inferior": sma - (num_std * std)
    }
    return resultado

def detectar_maximos_minimos(precios: pd.Series, ventana: int = 5) -> Dict:
    """
    Detecta máximos y mínimos locales comparando con periodos vecinos.
    """
    maximos = []
    minimos = []
    
    # Recorremos asegurando que haya suficientes elementos a la izquierda y derecha
    for i in range(ventana, len(precios) - ventana):
        sub_serie = precios.iloc[i - ventana : i + ventana + 1]
        precio_actual = precios.iloc[i]
        
        if precio_actual == sub_serie.max():
            maximos.append((precios.index[i], precio_actual))
        if precio_actual == sub_serie.min():
            minimos.append((precios.index[i], precio_actual))
            
    # Convertimos los resultados a Series de Pandas con su respectiva Fecha en el Índice
    idx_max, val_max = zip(*maximos) if maximos else ([], [])
    idx_min, val_min = zip(*minimos) if minimos else ([], [])
    
    return {
        "maximos": pd.Series(val_max, index=idx_max, dtype=float),
        "minimos": pd.Series(val_min, index=idx_min, dtype=float)
    }

def clasificar_tendencia(precios: pd.Series, ventana: int = 10) -> str:
    """
    Clasifica la tendencia actual basándose en la Media Móvil.
    """
    ma = media_movil(precios, ventana)
    
    if len(precios) < 2 or pd.isna(ma.iloc[-1]) or pd.isna(ma.iloc[-2]):
        return "LATERAL"
        
    precio_actual = precios.iloc[-1]
    ma_actual = ma.iloc[-1]
    ma_anterior = ma.iloc[-2]
    
    if precio_actual > ma_actual and ma_actual > ma_anterior:
        return "ALCISTA"
    elif precio_actual < ma_actual and ma_actual < ma_anterior:
        return "BAJISTA"
    else:
        return "LATERAL"

# =====================================================================
# PARTE 3: Sistema de Alertas
# =====================================================================

def generar_senales_trading(precios: pd.Series, ma_corta: int = 5, ma_larga: int = 20) -> pd.Series:
    """
    Genera señales de compra/venta basadas en cruces de medias móviles.
    """
    sma5 = media_movil(precios, ma_corta)
    sma20 = media_movil(precios, ma_larga)
    
    senales = pd.Series("MANTENER", index=precios.index)
    
    for i in range(1, len(precios)):
        # Verificar que existan datos suficientes de ambas medias
        if pd.isna(sma5.iloc[i-1]) or pd.isna(sma20.iloc[i-1]):
            continue
            
        # Cruce hacia arriba: Corta supera a larga
        if sma5.iloc[i-1] <= sma20.iloc[i-1] and sma5.iloc[i] > sma20.iloc[i]:
            senales.iloc[i] = "COMPRA"
        # Cruce hacia abajo: Corta cae por debajo de larga
        if sma5.iloc[i-1] >= sma20.iloc[i-1] and sma5.iloc[i] < sma20.iloc[i]:
            senales.iloc[i] = "VENTA"
            
    return senales

def alertas_precio(precios: pd.Series, umbral_cambio: float = 5.0) -> List[Dict]:
    """
    Genera alertas cuando hay cambios diarios significativos mayores al umbral.
    """
    rendimientos = calcular_rendimientos(precios)
    alertas = []
    
    for fecha, rend in rendimientos.dropna().items():
        if abs(rend) >= umbral_cambio:
            tipo = "SUBIDA" if rend > 0 else "CAIDA"
            alertas.append({
                "fecha": fecha.strftime('%Y-%m-%d'),
                "tipo": tipo,
                "cambio": float(rend)
            })
    return alertas

def clasificar_volatilidad(rendimientos: pd.Series) -> str:
    """
    Clasifica el nivel de volatilidad según la desviación estándar de los rendimientos.
    """
    vol = rendimientos.dropna().std()
    if pd.isna(vol):
        return "DESCONOCIDA"
        
    if vol < 1.0:
        return "BAJA"
    elif vol <= 3.0:
        return "MEDIA"
    elif vol <= 5.0:
        return "ALTA"
    else:
        return "MUY ALTA"

def generar_reporte_completo(precios: pd.Series, nombre_accion: str) -> Dict:
    """
    Integra todas las funciones anteriores para generar el reporte de inversión.
    """
    rendimientos = calcular_rendimientos(precios)
    senales = generar_senales_trading(precios)
    
    reporte = {
        "nombre": nombre_accion,
        "periodo": {
            "inicio": precios.index[0].strftime('%Y-%m-%d'),
            "fin": precios.index[-1].strftime('%Y-%m-%d'),
            "dias": int(precios.count())
        },
        "estadisticas": estadisticas_basicas(precios),
        "rendimientos": analisis_rendimientos(rendimientos),
        "tendencia": clasificar_tendencia(precios),
        "volatilidad": clasificar_volatilidad(rendimientos),
        "senal_actual": senales.iloc[-1],
        "alertas_recientes": alertas_precio(precios, umbral_cambio=5.0)
    }
    return reporte

# =====================================================================
# FUNCIONES DE VISUALIZACIÓN (Dadas en el enunciado)
# =====================================================================

def mostrar_reporte(reporte: Dict) -> None:
    print("=" * 70)
    print(f"           REPORTE DE ANÁLISIS: {reporte['nombre']}")
    print("=" * 70)
    
    periodo = reporte.get('periodo', {})
    print(f"\n📅 PERÍODO DE ANÁLISIS")
    print("-" * 40)
    print(f"Inicio: {periodo.get('inicio', 'N/A')}")
    print(f"Fin: {periodo.get('fin', 'N/A')}")
    print(f"Días analizados: {periodo.get('dias', 'N/A')}")
    
    stats = reporte.get('estadisticas', {})
    print(f"\n📊 ESTADÍSTICAS DE PRECIO")
    print("-" * 40)
    print(f"Precio actual:  ${stats.get('precio_actual', 0):,.2f}")
    print(f"Precio mínimo:  ${stats.get('precio_minimo', 0):,.2f}")
    print(f"Precio máximo:  ${stats.get('precio_maximo', 0):,.2f}")
    print(f"Precio promedio: ${stats.get('precio_promedio', 0):,.2f}")
    
    rend = reporte.get('rendimientos', {})
    print(f"\n📈 RENDIMIENTO")
    print("-" * 40)
    print(f"Rendimiento total: {rend.get('rendimiento_total', 0):+.2f}%")
    print(f"Rendimiento promedio diario: {rend.get('rendimiento_promedio', 0):+.3f}%")
    if rend.get('mejor_dia'):
        print(f"Mejor día: {rend['mejor_dia'][0]} ({rend['mejor_dia'][1]:+.2f}%)")
    if rend.get('peor_dia'):
        print(f"Peor día: {rend['peor_dia'][0]} ({rend['peor_dia'][1]:+.2f}%)")
    print(f"Días positivos: {rend.get('dias_positivos', 0)}")
    print(f"Días negativos: {rend.get('dias_negativos', 0)}")
    
    print(f"\n🎯 INDICADORES")
    print("-" * 40)
    print(f"Tendencia: {reporte.get('tendencia', 'N/A')}")
    print(f"Volatilidad: {reporte.get('volatilidad', 'N/A')}")
    print(f"Señal actual: {reporte.get('senal_actual', 'N/A')}")
    
    alertas = reporte.get('alertas_recientes', [])
    if alertas:
        print(f"\n⚠️ ALERTAS RECIENTES")
        print("-" * 40)
        for alerta in alertas[-5:]:
            emoji = "🔺" if alerta['tipo'] == 'SUBIDA' else "🔻"
            print(f"{emoji} {alerta['fecha']}: {alerta['tipo']} de {alerta['cambio']:+.2f}%")
    
    print("\n" + "=" * 70)

# =====================================================================
# BLOQUE PRINCIPAL: SIMULACIÓN DE DATOS Y PRUEBAS
# =====================================================================

if __name__ == "__main__":
    fechas = pd.date_range(start='2024-01-01', periods=60, freq='B')
    
    # 1. ACME Corp (Tendencia Alcista / Volatilidad Media)
    np.random.seed(42)
    rend_acme = np.random.normal(0.002, 0.02, 60)
    PRECIOS_ACCION = pd.Series((100 * np.cumprod(1 + rend_acme)).round(2), index=fechas, name='ACME Corp')
    
    # 2. VolatilTech (Alta volatilidad)
    np.random.seed(123)
    rend_volatil = np.random.normal(0, 0.05, 60)
    ACCION_VOLATIL = pd.Series((50 * np.cumprod(1 + rend_volatil)).round(2), index=fechas, name='VolatilTech')
    
    # 3. DeclineCorp (Tendencia Bajista)
    np.random.seed(123)
    rend_bajista = np.random.normal(-0.005, 0.015, 60)
    ACCION_BAJISTA = pd.Series((200 * np.cumprod(1 + rend_bajista)).round(2), index=fechas, name='DeclineCorp')

    print("PRUEBA DE FUNCIONES INDIVIDUALES")
    print("=" * 50)
    
    print("\n-- Estadísticas Básicas (ACME Corp) --")
    print(estadisticas_basicas(PRECIOS_ACCION))
    
    print("\n-- Rendimientos (primeros 5 días de ACME Corp) --")
    rendimientos = calcular_rendimientos(PRECIOS_ACCION)
    print(rendimientos.head())
    
    print("\nGENERANDO REPORTE COMPLETO...\n")
    reporte = generar_reporte_completo(PRECIOS_ACCION, "ACME Corp")
    mostrar_reporte(reporte)
    
    print("\n" + "=" * 70)
    print("         COMPARACIÓN DE ACCIONES")
    print("=" * 70)
    
    acciones = [
        (PRECIOS_ACCION, "ACME Corp"),
        (ACCION_VOLATIL, "VolatilTech"),
        (ACCION_BAJISTA, "DeclineCorp")
    ]
    
    for precios, nombre in acciones:
        rend = calcular_rendimientos(precios)
        print(f"\n{nombre}:")
        print(f"  Rendimiento Acumulado: {rend.sum():+.2f}%")
        print(f"  Volatilidad de la Acción: {clasificar_volatilidad(rend)}")
        print(f"  Tendencia de Mercado: {clasificar_tendencia(precios)}")
