import numpy as np

# Configuracion inicial

np.random.seed(42)

print("✅ NumPy cargado correctamente")
print(f"   Versión: {np.__version__}")


#                    GENERACIÓN DE DATOS


estaciones = ['Coyoacán', 'Azcapotzalco', 'Xochimilco', 'Tlalpan', 'Miguel Hidalgo']
n_estaciones = len(estaciones)
n_dias = 7
n_horas = 24

# ── Temperatura ──────────────────────────────────────────────────
temp_base = np.array([22, 24, 20, 19, 23])
hora_del_dia = np.arange(24)
variacion_diaria = 5 * np.sin((hora_del_dia - 6) * np.pi / 12)

temperatura = np.zeros((n_estaciones, n_dias, n_horas))
for i in range(n_estaciones):
    for d in range(n_dias):
        temperatura[i, d, :] = temp_base[i] + variacion_diaria + np.random.normal(0, 1.5, n_horas)

temperatura[1, 2, 10:14] = np.nan
temperatura[3, 5, 0:3]  = np.nan

# ── Humedad ───────────────────────────────────────────────────────
humedad_base = np.array([55, 45, 70, 65, 50])
variacion_humedad = -15 * np.sin((hora_del_dia - 6) * np.pi / 12)

humedad = np.zeros((n_estaciones, n_dias, n_horas))
for i in range(n_estaciones):
    for d in range(n_dias):
        humedad[i, d, :] = humedad_base[i] + variacion_humedad + np.random.normal(0, 5, n_horas)
humedad = np.clip(humedad, 20, 95)
humedad[0, 4, 15:18] = np.nan

# ── CO2 ───────────────────────────────────────────────────────────
co2_base = np.array([380, 420, 360, 350, 410])
patron_trafico = np.zeros(24)
patron_trafico[7:10]  = 30
patron_trafico[17:20] = 40
patron_trafico[12:14] = 15

co2 = np.zeros((n_estaciones, n_dias, n_horas))
for i in range(n_estaciones):
    for d in range(n_dias):
        co2[i, d, :] = co2_base[i] + patron_trafico + np.random.normal(0, 10, n_horas)
co2[:, 3, :] *= 1.15
co2[2, 1, 5:8] = np.nan

# ── Promedios diarios ─────────────────────────────────────────────
temp_promedio_diario    = np.nanmean(temperatura, axis=2)
humedad_promedio_diario = np.nanmean(humedad, axis=2)
co2_promedio_diario     = np.nanmean(co2, axis=2)

print("╔══════════════════════════════════════════════════════════════╗")
print("║              DATOS GENERADOS EXITOSAMENTE                    ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║     temperatura     : shape {temperatura.shape}               ║")
print(f"║    humedad         : shape {humedad.shape}               ║")
print(f"║    co2             : shape {co2.shape}               ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║    temp_promedio_diario    : shape {temp_promedio_diario.shape}            ║")
print(f"║    humedad_promedio_diario : shape {humedad_promedio_diario.shape}            ║")
print(f"║    co2_promedio_diario     : shape {co2_promedio_diario.shape}            ║")
print("╚══════════════════════════════════════════════════════════════╝")
print("\n Estaciones:", estaciones)


# ══════════════════════════════════════════════════════════════════
#           PARTE 1: EXPLORACIÓN DE ARRAYS (25 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("   PARTE 1: EXPLORACIÓN DE ARRAYS")
print("═"*60)

# ── Ejercicio 1.1: Inspección ─────────────────────────────────────
n_dimensiones    = temperatura.ndim
forma            = temperatura.shape
total_elementos  = temperatura.size
tipo_datos       = temperatura.dtype
memoria_bytes    = temperatura.nbytes

print("\n PROPIEDADES DEL ARRAY TEMPERATURA")
print("─" * 40)
print(f"Dimensiones: {n_dimensiones}D")
print(f"Forma: {forma}")
print(f"  → {forma[0]} estaciones")
print(f"  → {forma[1]} días")
print(f"  → {forma[2]} horas por día")
print(f"Total de mediciones: {total_elementos:,}")
print(f"Tipo de datos: {tipo_datos}")
print(f"Memoria: {memoria_bytes:,} bytes ({memoria_bytes/1024:.2f} KB)")

# ── Ejercicio 1.2: Indexación ─────────────────────────────────────
temp_coyoacan_d1_12h = temperatura[0, 0, 12]
print(f"\n Coyoacán, Día 1, 12:00h: {temp_coyoacan_d1_12h:.1f}°C")

temp_xochimilco_d3 = temperatura[2, 2, :]
print(f"\n Xochimilco, Día 3 (24 horas):")
print(f"   Primeras 6 horas: {temp_xochimilco_d3[:6].round(1)}")

temp_mh_7dias = temp_promedio_diario[4, :]
print(f"\n Miguel Hidalgo - Promedio por día:")
print(f"   {temp_mh_7dias.round(1)}")

ultimo_co2 = co2[-1, -1, -1]
print(f"\n Último CO2 registrado: {ultimo_co2:.1f} ppm")

# ── Ejercicio 1.3: Slicing ────────────────────────────────────────
temp_tardes = temperatura[:, :, 12:18]
print(f"\n Temperaturas de tardes (12-17h)")
print(f"   Shape: {temp_tardes.shape}")

humedad_subset = humedad[:3, -3:, :]
print(f"\n Subset de humedad")
print(f"   Shape: {humedad_subset.shape}")

co2_mañanas_pares = co2[::2, :, 6:12]
print(f"\n🏭 CO2 mañanas (estaciones pares)")
print(f"   Shape: {co2_mañanas_pares.shape}")

temp_inverso = temperatura[:, ::-1, :]
print(f"\n Temperatura días invertidos")
print(f"   Shape: {temp_inverso.shape}")


# ══════════════════════════════════════════════════════════════════
#          PARTE 2: ESTADÍSTICAS BÁSICAS (25 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("   PARTE 2: ESTADÍSTICAS BÁSICAS")
print("═"*60)

# ── Ejercicio 2.1: Estadísticas globales ──────────────────────────
temp_promedio = np.nanmean(temperatura)
temp_maxima   = np.nanmax(temperatura)
temp_minima   = np.nanmin(temperatura)
temp_std      = np.nanstd(temperatura)
temp_rango    = temp_maxima - temp_minima

print("╔══════════════════════════════════════════════════════════════╗")
print("║           ESTADÍSTICAS GLOBALES DE TEMPERATURA               ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  Promedio:     {temp_promedio:>6.2f} °C                              ║")
print(f"║  Máxima:       {temp_maxima:>6.2f} °C                              ║")
print(f"║  Mínima:       {temp_minima:>6.2f} °C                              ║")
print(f"║  Desv. Est.:   {temp_std:>6.2f} °C                              ║")
print(f"║  Rango:        {temp_rango:>6.2f} °C                              ║")
print("╚══════════════════════════════════════════════════════════════╝")

# ── Ejercicio 2.2: Estadísticas por eje ───────────────────────────
temp_por_estacion = np.nanmean(temperatura, axis=(1, 2))

print("\n TEMPERATURA PROMEDIO POR ESTACIÓN")
print("─" * 40)
for i, est in enumerate(estaciones):
    print(f"   {est:15s}: {temp_por_estacion[i]:5.1f} °C")

humedad_por_hora = np.nanmean(humedad, axis=(0, 1))

print("\n HUMEDAD PROMEDIO POR HORA")
print("─" * 40)
print("   Hora │ Humedad")
for h in [0, 6, 12, 18]:
    print(f"   {h:02d}:00 │ {humedad_por_hora[h]:5.1f}%")

co2_max_por_dia = np.nanmax(co2, axis=(0, 2))

print("\n CO2 MÁXIMO POR DÍA")
print("─" * 40)
for d in range(n_dias):
    print(f"   Día {d+1}: {co2_max_por_dia[d]:6.1f} ppm")


# ══════════════════════════════════════════════════════════════════
#         PARTE 3: OPERACIONES VECTORIZADAS (25 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("   PARTE 3: OPERACIONES VECTORIZADAS")
print("═"*60)

# ── Ejercicio 3.1: Conversiones ───────────────────────────────────
temperatura_fahrenheit = temperatura * 9/5 + 32

print(" TEMPERATURA EN FAHRENHEIT")
print(f"   Promedio: {np.nanmean(temperatura_fahrenheit):.1f} °F")
print(f"   Máxima:   {np.nanmax(temperatura_fahrenheit):.1f} °F")
print(f"   Mínima:   {np.nanmin(temperatura_fahrenheit):.1f} °F")

temperatura_kelvin = temperatura + 273.15

print(f"\n TEMPERATURA EN KELVIN")
print(f"   Promedio: {np.nanmean(temperatura_kelvin):.1f} K")

humedad_min = np.nanmin(humedad)
humedad_max = np.nanmax(humedad)
humedad_normalizada = (humedad - humedad_min) / (humedad_max - humedad_min)

print(f"\n HUMEDAD NORMALIZADA [0-1]")
print(f"   Promedio: {np.nanmean(humedad_normalizada):.3f}")
print(f"   Min:      {np.nanmin(humedad_normalizada):.3f}")
print(f"   Max:      {np.nanmax(humedad_normalizada):.3f}")

# ── Ejercicio 3.2: Índice de Confort Térmico ──────────────────────
ict = temperatura + 0.05 * humedad

print("\n ÍNDICE DE CONFORT TÉRMICO (ICT)")
print("─" * 45)
print(f"   Shape del array ICT: {ict.shape}")
print(f"   ICT promedio: {np.nanmean(ict):.2f}")
print(f"   ICT máximo:   {np.nanmax(ict):.2f}")
print(f"   ICT mínimo:   {np.nanmin(ict):.2f}")

n_frio        = int(np.sum(ict < 20))
n_confortable = int(np.sum((ict >= 20) & (ict < 25)))
n_calido      = int(np.sum((ict >= 25) & (ict < 30)))
n_muy_caluroso = int(np.sum(ict >= 30))
n_validas     = int(np.sum(~np.isnan(ict)))

print("\n DISTRIBUCIÓN DE CONDICIONES")
print("─" * 45)
print(f"      Frío (<20):           {n_frio:5d} ({100*n_frio/n_validas:5.1f}%)")
print(f"     Confortable (20-25):  {n_confortable:5d} ({100*n_confortable/n_validas:5.1f}%)")
print(f"      Cálido (25-30):       {n_calido:5d} ({100*n_calido/n_validas:5.1f}%)")
print(f"     Muy caluroso (≥30):   {n_muy_caluroso:5d} ({100*n_muy_caluroso/n_validas:5.1f}%)")
print(f"   ────────────────────────────────────────")
print(f"   Total válidas:           {n_validas:5d}")


# ══════════════════════════════════════════════════════════════════
#            PARTE 4: ANÁLISIS AVANZADO (25 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("   PARTE 4: ANÁLISIS AVANZADO")
print("═"*60)

# ── Ejercicio 4.1: Detección de anomalías ─────────────────────────
co2_media = np.nanmean(co2)
co2_std   = np.nanstd(co2)
limite_inferior = co2_media - 2 * co2_std
limite_superior = co2_media + 2 * co2_std

print("\n🏭 ANÁLISIS DE ANOMALÍAS EN CO2")
print("─" * 45)
print(f"   Media CO2:       {co2_media:.1f} ppm")
print(f"   Desv. Est.:      {co2_std:.1f} ppm")
print(f"   Límite inferior: {limite_inferior:.1f} ppm")
print(f"   Límite superior: {limite_superior:.1f} ppm")

mascara_anomalias = (~np.isnan(co2)) & ((co2 < limite_inferior) | (co2 > limite_superior))
n_anomalias   = int(np.sum(mascara_anomalias))
valores_anomalos = co2[mascara_anomalias]

print(f"\n   ANOMALÍAS DETECTADAS: {n_anomalias}")
if n_anomalias > 0:
    print(f"   Valores: {valores_anomalos[:10].round(1)}")
    if n_anomalias > 10:
        print(f"   ... y {n_anomalias - 10} más")

# ── Ejercicio 4.2: Análisis de contingencia ambiental ─────────────
DIA_CONTINGENCIA = 3

co2_contingencia   = co2[:, DIA_CONTINGENCIA, :]
dias_normales      = [0, 1, 2, 4, 5, 6]
co2_dias_normales  = co2[:, dias_normales, :]

promedio_contingencia = float(np.nanmean(co2_contingencia))
promedio_normal       = float(np.nanmean(co2_dias_normales))
incremento_porcentual = (promedio_contingencia - promedio_normal) / promedio_normal * 100

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║           ANÁLISIS DE CONTINGENCIA AMBIENTAL                 ║")
print("║                        Día 4                                 ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  CO2 promedio día contingencia: {promedio_contingencia:>7.1f} ppm              ║")
print(f"║  CO2 promedio días normales:    {promedio_normal:>7.1f} ppm              ║")
print(f"║  Incremento:                    {incremento_porcentual:>7.1f} %               ║")
print("╚══════════════════════════════════════════════════════════════╝")

co2_por_estacion_contingencia = np.nanmean(co2_contingencia, axis=1)
co2_por_estacion_normal       = np.nanmean(co2_dias_normales, axis=(1, 2))

incremento_por_estacion = (
    (co2_por_estacion_contingencia - co2_por_estacion_normal) /
    co2_por_estacion_normal * 100
)
idx_mas_afectada = int(np.argmax(incremento_por_estacion))

print("\n IMPACTO POR ESTACIÓN")
print("─" * 50)
for i, est in enumerate(estaciones):
    barra = "█" * int(incremento_por_estacion[i] / 2)
    print(f"   {est:15s}: +{incremento_por_estacion[i]:5.1f}% {barra}")

print(f"\n  Estación más afectada: {estaciones[idx_mas_afectada]}")


# ══════════════════════════════════════════════════════════════════
#          BONUS: REPORTE EJECUTIVO (+10 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("   BONUS: REPORTE EJECUTIVO")
print("═"*60)

temp_prom_por_est  = np.nanmean(temperatura, axis=(1, 2))
hum_prom_por_est   = np.nanmean(humedad, axis=(1, 2))
co2_prom_por_est   = np.nanmean(co2, axis=(1, 2))

idx_mas_calurosa = int(np.argmax(temp_prom_por_est))
estacion_mas_calurosa = estaciones[idx_mas_calurosa]

idx_mas_humeda = int(np.argmax(hum_prom_por_est))
estacion_mas_humeda = estaciones[idx_mas_humeda]

idx_mejor_aire = int(np.argmin(co2_prom_por_est))
estacion_mejor_aire = estaciones[idx_mejor_aire]

temp_por_hora    = np.nanmean(temperatura, axis=(0, 1))
hora_mas_calurosa = int(np.argmax(temp_por_hora))

co2_por_hora = np.nanmean(co2, axis=(0, 1))
hora_peor_aire = int(np.argmax(co2_por_hora))

nan_temperatura = int(np.sum(np.isnan(temperatura)))
nan_humedad     = int(np.sum(np.isnan(humedad)))
nan_co2         = int(np.sum(np.isnan(co2)))
total_nan       = nan_temperatura + nan_humedad + nan_co2

print("")
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║                                                                      ║")
print("║              METEOSENSE - REPORTE EJECUTIVO SEMANAL              ║")
print("║                        CDMX - Semana de Análisis                     ║")
print("║                                                                      ║")
print("╠══════════════════════════════════════════════════════════════════════╣")
print("║                                                                      ║")
print("║   RESUMEN DE CONDICIONES                                           ║")
print("║  ─────────────────────────────────────────────────────────────────   ║")
print(f"║      Temperatura promedio:    {np.nanmean(temperatura):>5.1f} °C                        ║")
print(f"║     Humedad promedio:         {np.nanmean(humedad):>5.1f} %                         ║")
print(f"║      CO2 promedio:            {np.nanmean(co2):>6.1f} ppm                       ║")
print("║                                                                      ║")
print("║    RANKINGS                                                         ║")
print("║  ─────────────────────────────────────────────────────────────────   ║")
print(f"║      Estación más calurosa:   {estacion_mas_calurosa:15s}                  ║")
print(f"║      Estación más húmeda:     {estacion_mas_humeda:15s}                  ║")
print(f"║      Mejor calidad de aire:   {estacion_mejor_aire:15s}                  ║")
print("║                                                                      ║")
print("║    PATRONES TEMPORALES                                              ║")
print("║  ─────────────────────────────────────────────────────────────────   ║")
print(f"║       Hora más calurosa:       {hora_mas_calurosa:02d}:00 hrs                          ║")
print(f"║      Hora con más CO2:         {hora_peor_aire:02d}:00 hrs                          ║")
print("║                                                                      ║")
print("║     CALIDAD DE DATOS                                                ║")
print("║  ─────────────────────────────────────────────────────────────────   ║")
print(f"║    Valores faltantes totales:  {total_nan:4d}                                 ║")
print(f"║      - Temperatura: {nan_temperatura:3d}                                            ║")
print(f"║      - Humedad:     {nan_humedad:3d}                                            ║")
print(f"║      - CO2:         {nan_co2:3d}                                            ║")
print("║                                                                      ║")
print("╚══════════════════════════════════════════════════════════════════════╝")
print("")
