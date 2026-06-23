import numpy as np

#                    CONFIGURACIÓN INICIAL

np.set_printoptions(precision=2, suppress=True)

print(" NumPy cargado correctamente")
print(f"   Versión: {np.__version__}")

#                 GENERACIÓN DE DATOS DE TRANSACCIONES

np.random.seed(2024)

categorias = ['Supermercados', 'Restaurantes', 'Gasolineras', 'Tiendas_Online', 'Entretenimiento']
n_categorias = len(categorias)

params_categorias = {
    'Supermercados':   (800,  400),
    'Restaurantes':    (350,  150),
    'Gasolineras':     (700,  250),
    'Tiendas_Online':  (1200, 800),
    'Entretenimiento': (200,  100),
}

n_transacciones_por_cat = 500

transacciones = {}
ids_transaccion = {}

for i, cat in enumerate(categorias):
    media, std = params_categorias[cat]
    montos = np.random.normal(media, std, n_transacciones_por_cat)
    montos = np.maximum(montos, 10)

    n_anomalias_altas = np.random.randint(8, 15)
    n_anomalias_bajas = np.random.randint(5, 10)

    indices_altas = np.random.choice(n_transacciones_por_cat, n_anomalias_altas, replace=False)
    montos[indices_altas] = media + np.random.uniform(4, 8, n_anomalias_altas) * std

    indices_bajas = np.random.choice(
        [j for j in range(n_transacciones_por_cat) if j not in indices_altas],
        n_anomalias_bajas, replace=False
    )
    montos[indices_bajas] = np.random.uniform(1, 15, n_anomalias_bajas)

    transacciones[cat] = montos
    ids_transaccion[cat] = np.arange(i * 1000 + 1, i * 1000 + n_transacciones_por_cat + 1)

montos_matriz   = np.array([transacciones[cat] for cat in categorias])
todos_montos    = np.concatenate([transacciones[cat] for cat in categorias])
todas_categorias = np.concatenate([[cat] * n_transacciones_por_cat for cat in categorias])
todos_ids       = np.concatenate([ids_transaccion[cat] for cat in categorias])

print("╔══════════════════════════════════════════════════════════════════╗")
print("║              DATOS DE TRANSACCIONES GENERADOS                    ║")
print("╠══════════════════════════════════════════════════════════════════╣")
print(f"║    montos_matriz    : shape {montos_matriz.shape}                      ║")
print(f"║     (filas=categorías, columnas=transacciones)                  ║")
print(f"║                                                                  ║")
print(f"║    todos_montos     : {len(todos_montos):,} transacciones totales          ║")
print(f"║     todas_categorias : {len(todas_categorias):,} etiquetas                    ║")
print(f"║    todos_ids        : {len(todos_ids):,} identificadores                ║")
print("╚══════════════════════════════════════════════════════════════════╝")

print("\n  Categorías disponibles:")
for i, cat in enumerate(categorias):
    media, std = params_categorias[cat]
    print(f"   {i}: {cat:20s} (μ=${media:,}, σ=${std})")


#        PARTE 1: ANÁLISIS ESTADÍSTICO POR CATEGORÍA (30 pts)

print("\n" + "═"*80)
print("   PARTE 1: ANÁLISIS ESTADÍSTICO POR CATEGORÍA")
print("═"*80)

# ── Ejercicio 1.1: Estadísticas descriptivas ──────────────────────
print("\n  ESTADÍSTICAS DESCRIPTIVAS POR CATEGORÍA")
print("═" * 80)

medias   = np.zeros(n_categorias)
medianas = np.zeros(n_categorias)
stds     = np.zeros(n_categorias)
minimos  = np.zeros(n_categorias)
maximos  = np.zeros(n_categorias)

for i, cat in enumerate(categorias):
    datos = montos_matriz[i]
    medias[i]   = np.mean(datos)
    medianas[i] = np.median(datos)
    stds[i]     = np.std(datos)
    minimos[i]  = np.min(datos)
    maximos[i]  = np.max(datos)

print(f"\n{'Categoría':<20} {'Media':>12} {'Mediana':>12} {'Std':>12} {'Mín':>10} {'Máx':>10}")
print("─" * 80)
for i, cat in enumerate(categorias):
    print(f"{cat:<20} ${medias[i]:>10,.2f} ${medianas[i]:>10,.2f} "
          f"${stds[i]:>10,.2f} ${minimos[i]:>8,.2f} ${maximos[i]:>8,.2f}")

# ── Ejercicio 1.2: Cuartiles e IQR ───────────────────────────────
print("\n  CUARTILES E IQR POR CATEGORÍA")
print("═" * 80)

Q1_arr  = np.zeros(n_categorias)
Q2_arr  = np.zeros(n_categorias)
Q3_arr  = np.zeros(n_categorias)
IQR_arr = np.zeros(n_categorias)

for i, cat in enumerate(categorias):
    datos = montos_matriz[i]
    Q1_arr[i]  = np.percentile(datos, 25)
    Q2_arr[i]  = np.percentile(datos, 50)
    Q3_arr[i]  = np.percentile(datos, 75)
    IQR_arr[i] = Q3_arr[i] - Q1_arr[i]

print(f"\n{'Categoría':<20} {'Q1 (25%)':>12} {'Q2 (50%)':>12} {'Q3 (75%)':>12} {'IQR':>12}")
print("─" * 72)
for i, cat in enumerate(categorias):
    print(f"{cat:<20} ${Q1_arr[i]:>10,.2f} ${Q2_arr[i]:>10,.2f} "
          f"${Q3_arr[i]:>10,.2f} ${IQR_arr[i]:>10,.2f}")

# ── Ejercicio 1.3: Límites para outliers ─────────────────────────
print("\n LÍMITES PARA DETECCIÓN DE OUTLIERS (Método IQR)")
print("═" * 80)

FACTOR_IQR = 1.5

limites_inf = np.zeros(n_categorias)
limites_sup = np.zeros(n_categorias)

for i in range(n_categorias):
    limites_inf[i] = Q1_arr[i] - FACTOR_IQR * IQR_arr[i]
    limites_sup[i] = Q3_arr[i] + FACTOR_IQR * IQR_arr[i]

print(f"\n{'Categoría':<20} {'Límite Inf':>15} {'Límite Sup':>15} {'Rango Válido':>20}")
print("─" * 75)
for i, cat in enumerate(categorias):
    lim_inf_real = max(0, limites_inf[i])
    rango = f"${lim_inf_real:,.0f} - ${limites_sup[i]:,.0f}"
    print(f"{cat:<20} ${limites_inf[i]:>13,.2f} ${limites_sup[i]:>13,.2f} {rango:>20}")


# ══════════════════════════════════════════════════════════════════
#          PARTE 2: DETECCIÓN DE OUTLIERS CON IQR (25 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("   PARTE 2: DETECCIÓN DE OUTLIERS CON IQR")
print("═"*80)

# ── Ejercicio 2.1: Identificar outliers por categoría ─────────────
print("\n DETECCIÓN DE TRANSACCIONES ANÓMALAS (Método IQR)")
print("═" * 80)

outliers_iqr    = {}
n_outliers_iqr  = np.zeros(n_categorias, dtype=int)

for i, cat in enumerate(categorias):
    datos = montos_matriz[i]
    ids   = ids_transaccion[cat]

    mascara_outliers = (datos < limites_inf[i]) | (datos > limites_sup[i])

    mascara_inf = datos < limites_inf[i]
    mascara_sup = datos > limites_sup[i]

    outliers_iqr[cat] = {
        'ids':          ids[mascara_outliers],
        'montos':       datos[mascara_outliers],
        'n_total':      int(np.sum(mascara_outliers)),
        'n_inferiores': int(np.sum(mascara_inf)),
        'n_superiores': int(np.sum(mascara_sup)),
    }
    n_outliers_iqr[i] = int(np.sum(mascara_outliers))

print(f"\n{'Categoría':<20} {'Total Trans.':>12} {'Outliers':>10} {'% Anomalías':>12} {'Inf.':>8} {'Sup.':>8}")
print("─" * 75)
for i, cat in enumerate(categorias):
    pct  = (n_outliers_iqr[i] / n_transacciones_por_cat) * 100
    info = outliers_iqr[cat]
    print(f"{cat:<20} {n_transacciones_por_cat:>12,} {info['n_total']:>10} "
          f"{pct:>11.1f}% {info['n_inferiores']:>8} {info['n_superiores']:>8}")

print(f"\n  Total de outliers detectados: {np.sum(n_outliers_iqr)}")

# ── Ejercicio 2.2: Análisis de outliers detectados ────────────────
print("\n  ANÁLISIS DETALLADO DE OUTLIERS (Método IQR)")
print("═" * 80)

for cat in categorias:
    info = outliers_iqr[cat]
    if info['n_total'] > 0:
        montos_out = info['montos']

        monto_min_outlier     = np.min(montos_out)
        monto_max_outlier     = np.max(montos_out)
        monto_promedio_outlier = np.mean(montos_out)

        print(f"\n   {cat}")
        print(f"   Outliers detectados: {info['n_total']}")
        print(f"   Monto mínimo outlier: ${monto_min_outlier:,.2f}")
        print(f"   Monto máximo outlier: ${monto_max_outlier:,.2f}")
        print(f"   Monto promedio outlier: ${monto_promedio_outlier:,.2f}")

        if info['n_superiores'] > 0:
            idx_ordenados = np.argsort(montos_out)[::-1]
            print(f"   Top 3 montos más altos:")
            for j in range(min(3, len(idx_ordenados))):
                idx = idx_ordenados[j]
                print(f"      - ID {info['ids'][idx]}: ${montos_out[idx]:,.2f}")


# ══════════════════════════════════════════════════════════════════
#       PARTE 3: DETECCIÓN DE OUTLIERS CON Z-SCORE (25 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("   PARTE 3: DETECCIÓN DE OUTLIERS CON Z-SCORE")
print("═"*80)

# ── Ejercicio 3.1: Calcular Z-Scores ─────────────────────────────
print("\n  CÁLCULO DE Z-SCORES POR CATEGORÍA")
print("═" * 80)

UMBRAL_ZSCORE   = 3
zscores_matriz  = np.zeros_like(montos_matriz)

for i, cat in enumerate(categorias):
    datos     = montos_matriz[i]
    media_cat = np.mean(datos)
    std_cat   = np.std(datos)
    zscores_matriz[i] = (datos - media_cat) / std_cat

print(f"\n{'Categoría':<20} {'Media Z':>10} {'Std Z':>10} {'Min Z':>10} {'Max Z':>10}")
print("─" * 65)
for i, cat in enumerate(categorias):
    zs = zscores_matriz[i]
    print(f"{cat:<20} {np.mean(zs):>10.4f} {np.std(zs):>10.4f} "
          f"{np.min(zs):>10.2f} {np.max(zs):>10.2f}")

print(f"\n  Nota: La media de Z-scores debe ser ~0 y la std ~1")

# ── Ejercicio 3.2: Detectar outliers con Z-Score ─────────────────
print(f"\n  DETECCIÓN DE OUTLIERS CON Z-SCORE (umbral = {UMBRAL_ZSCORE})")
print("═" * 80)

outliers_zscore   = {}
n_outliers_zscore = np.zeros(n_categorias, dtype=int)

for i, cat in enumerate(categorias):
    datos   = montos_matriz[i]
    zscores = zscores_matriz[i]
    ids     = ids_transaccion[cat]

    mascara_outliers_z = np.abs(zscores) > UMBRAL_ZSCORE

    mascara_z_neg = zscores < -UMBRAL_ZSCORE
    mascara_z_pos = zscores >  UMBRAL_ZSCORE

    outliers_zscore[cat] = {
        'ids':     ids[mascara_outliers_z],
        'montos':  datos[mascara_outliers_z],
        'zscores': zscores[mascara_outliers_z],
        'n_total': int(np.sum(mascara_outliers_z)),
        'n_bajos': int(np.sum(mascara_z_neg)),
        'n_altos': int(np.sum(mascara_z_pos)),
    }
    n_outliers_zscore[i] = int(np.sum(mascara_outliers_z))

print(f"\n{'Categoría':<20} {'Total Trans.':>12} {'Outliers':>10} {'% Anomalías':>12} {'Z<-3':>8} {'Z>3':>8}")
print("─" * 75)
for i, cat in enumerate(categorias):
    pct  = (n_outliers_zscore[i] / n_transacciones_por_cat) * 100
    info = outliers_zscore[cat]
    print(f"{cat:<20} {n_transacciones_por_cat:>12,} {info['n_total']:>10} "
          f"{pct:>11.1f}% {info['n_bajos']:>8} {info['n_altos']:>8}")

print(f"\n  Total de outliers detectados (Z-Score): {np.sum(n_outliers_zscore)}")


# ══════════════════════════════════════════════════════════════════
#         PARTE 4: COMPARACIÓN Y REPORTE FINAL (20 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*80)
print("   PARTE 4: COMPARACIÓN Y REPORTE FINAL")
print("═"*80)

# ── Ejercicio 4.1: Comparar métodos ──────────────────────────────
print("\n  COMPARACIÓN DE MÉTODOS DE DETECCIÓN")
print("═" * 80)

total_iqr    = int(np.sum(n_outliers_iqr))
total_zscore = int(np.sum(n_outliers_zscore))

print(f"\n  RESUMEN GLOBAL:")
print(f"   Método IQR:     {total_iqr} outliers detectados")
print(f"   Método Z-Score: {total_zscore} outliers detectados")

print(f"\n{'Categoría':<20} {'IQR':>10} {'Z-Score':>10} {'Diferencia':>12} {'Coincidencia':>15}")
print("─" * 72)

for i, cat in enumerate(categorias):
    n_iqr = n_outliers_iqr[i]
    n_zs  = n_outliers_zscore[i]
    diff  = n_iqr - n_zs

    ids_iqr    = set(outliers_iqr[cat]['ids'])
    ids_zscore = set(outliers_zscore[cat]['ids'])
    coincidentes = len(ids_iqr & ids_zscore)

    print(f"{cat:<20} {n_iqr:>10} {n_zs:>10} {diff:>+12} {coincidentes:>15}")

# ── Ejercicio 4.2: Reporte de transacciones sospechosas ───────────
print("\n╔══════════════════════════════════════════════════════════════════════════╗")
print("║                                                                          ║")
print("║          SECUREBANK - REPORTE DE TRANSACCIONES SOSPECHOSAS             ║")
print("║                                                                          ║")
print("╠══════════════════════════════════════════════════════════════════════════╣")
print("║                                                                          ║")
print("║     ALTA PRIORIDAD (detectadas por ambos métodos)                       ║")
print("║  ─────────────────────────────────────────────────────────────────────   ║")

total_alta_prioridad = 0

for cat in categorias:
    ids_iqr    = set(outliers_iqr[cat]['ids'])
    ids_zscore = set(outliers_zscore[cat]['ids'])

    ids_ambos = ids_iqr & ids_zscore   # intersección

    if len(ids_ambos) > 0:
        total_alta_prioridad += len(ids_ambos)
        cat_idx = categorias.index(cat)

        for trans_id in list(ids_ambos)[:3]:
            idx    = np.where(ids_transaccion[cat] == trans_id)[0][0]
            monto  = montos_matriz[cat_idx, idx]
            zscore = zscores_matriz[cat_idx, idx]
            print(f"║    ID {trans_id}: {cat:15s} ${monto:>10,.2f} (Z={zscore:+.2f}){'':>10}║")

# Estadísticas finales
total_transacciones = n_categorias * n_transacciones_por_cat

ids_iqr_todos    = set(np.concatenate([outliers_iqr[c]['ids']    for c in categorias]))
ids_zscore_todos = set(np.concatenate([outliers_zscore[c]['ids'] for c in categorias]))
total_outliers_unicos = len(ids_iqr_todos | ids_zscore_todos)   # unión

pct_anomalias = (total_outliers_unicos / total_transacciones) * 100

print("║                                                                          ║")
print(f"║    RESUMEN EJECUTIVO                                                    ║")
print("║  ─────────────────────────────────────────────────────────────────────   ║")
print(f"║    Total transacciones analizadas:    {total_transacciones:>6,}{'':>25}║")
print(f"║    Transacciones sospechosas:         {total_outliers_unicos:>6,} ({pct_anomalias:.1f}%){'':>17}║")
print(f"║    Alta prioridad (ambos métodos):    {total_alta_prioridad:>6,}{'':>25}║")
print("║                                                                          ║")
print("╚══════════════════════════════════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════════════
#              BONUS: ANÁLISIS DE CORRELACIÓN (+10 pts)
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*70)
print("   BONUS: ANÁLISIS DE CORRELACIÓN ENTRE CATEGORÍAS")
print("═"*70)

print("\n  ANÁLISIS DE CORRELACIÓN ENTRE CATEGORÍAS")
print("═" * 70)

matriz_correlacion = np.corrcoef(montos_matriz)

print(f"\n{'':>18}", end='')
for cat in categorias:
    print(f"{cat[:8]:>10}", end='')
print()
print("─" * 70)

for i, cat in enumerate(categorias):
    print(f"{cat:<18}", end='')
    for j in range(n_categorias):
        valor = matriz_correlacion[i, j]
        if i == j:
            print(f"{'1.00':>10}", end='')
        else:
            print(f"{valor:>10.3f}", end='')
    print()

print(f"\n  CORRELACIONES MÁS FUERTES:")
print("─" * 50)

max_corr = 0.0
par_max  = ('', '')

for i in range(n_categorias):
    for j in range(i + 1, n_categorias):
        if abs(matriz_correlacion[i, j]) > abs(max_corr):
            max_corr = matriz_correlacion[i, j]
            par_max  = (categorias[i], categorias[j])

print(f"   Mayor correlación: {par_max[0]} ↔ {par_max[1]}")
print(f"   Valor: {max_corr:.4f}")
