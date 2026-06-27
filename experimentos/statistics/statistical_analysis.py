# ANALISIS ESTADISTICO DE RESULTADOS RAGAS

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

CARPETA_RESULTADOS = "data"

# Mapeo
ARCHIVOS = {
    "resultados_ragas_CS-500":          "CS-500",
    "resultados_ragas_CS-1000(BASELINE)": "CS-1000",
    "resultados_ragas_CS-2000":         "CS-2000",
    "resultados_ragas_RRR-AD-300":      "RRR-AD-300",
    "resultados_ragas_RRR-GE":          "RRR-GE",
    "resultados_ragas_RRR-AD":          "RRR-AD",
    "resultados_ragas_H-5050":          "H-50/50",
    "resultados_ragas_H-3070":          "H-30/70",
    "resultados_ragas_H-7030":          "H-70/30",
    "resultados_ragas_META-B":          "META-B",
    "resultados_ragas_MQ-B4x3":        "MQ-B4x3",
    "resultados_ragas_MQ-B3x4":        "MQ-B3x4",
}

BASELINE = "CS-1000"

# Grupos para Friedman
GRUPOS = {
    "Chunk Size":    ["CS-500", "CS-1000", "CS-2000"],
    "RRR":           ["RRR-AD-300", "RRR-GE", "RRR-AD"],
    "Hybrid":        ["H-50/50", "H-30/70", "H-70/30"],
}

# Metricas a evaluar
METRICAS = ["RAGAS Faithfulness", "RAGAS ResponseRelevancy", "RAGAS ContextPrecision"]

METRICAS_CORTAS = {
    "RAGAS Faithfulness": "Faithfulness",
    "RAGAS ResponseRelevancy": "Relevancy",
    "RAGAS ContextPrecision": "CtxPrecision",
}

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def rank_biserial_r(x, y):
    """
    Calcula el rank-biserial correlation para datos pareados
    r positivo = x > y, r negativo = y > x
    """
    diff = x - y
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(diff))
    r_plus = np.sum(ranks[diff > 0])
    r_minus = np.sum(ranks[diff < 0])
    r = (r_plus - r_minus) / (n * (n + 1) / 2)
    return r


def interpret_effect(r):
    """
    Interpretacion del tamano del efecto (rank-biserial r)
    Basado en los umbrales de Cohen adaptados
    """
    r_abs = abs(r)
    if r_abs < 0.1:
        return "insignificante"
    elif r_abs < 0.3:
        return "pequeno"
    elif r_abs < 0.5:
        return "mediano"
    else:
        return "grande"


def holm_correction(p_values):
    """
    Correccion de Holm-Bonferroni para comparaciones multiples
    Entrada: lista de tuplas (nombre, p_value)
    Salida: lista de tuplas (nombre, p_original, p_corregido)
    """
    n = len(p_values)
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    corrected = []
    for i, (name, p) in enumerate(sorted_pvals):
        p_corr = min(p * (n - i), 1.0)
        corrected.append((name, p, p_corr))

    # Asegurar monotonia: cada p corregido >= el anterior
    for i in range(1, len(corrected)):
        if corrected[i][2] < corrected[i-1][2]:
            corrected[i] = (corrected[i][0], corrected[i][1], corrected[i-1][2])

    return corrected


def significance_stars(p):
    """Estrellas de significancia"""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def pair_stats(c1, c2, metrica, datos_dict):
    """
    Funcion para calcular Wilcoxon y tamano de efecto entre dos configuraciones
    """
    comunes = sorted(set(datos_dict[c1].index) & set(datos_dict[c2].index))
    
    x = datos_dict[c1].loc[comunes, metrica].values
    y = datos_dict[c2].loc[comunes, metrica].values

    # Eliminar pares con NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 10:
        return None

    try:
        stat_w, p_w = stats.wilcoxon(x_clean, y_clean, alternative="two-sided")
        r = rank_biserial_r(x_clean, y_clean)
        return x_clean, y_clean, stat_w, p_w, r
    except Exception as e:
        print(f"  {c1} vs {c2} / {METRICAS_CORTAS[metrica]}: ERROR - {e}")
        return None


# ==========================================
# CARGA DE DATOS
# ==========================================

datos = {}
for archivo_base, config_name in ARCHIVOS.items():
    filepath = os.path.join(CARPETA_RESULTADOS, archivo_base + ".xlsx")
    if not os.path.exists(filepath):
        print(f"No encontrado: {filepath}")
        continue

    df = pd.read_excel(filepath, engine="openpyxl")

    # Agregacion por pregunta: mediana de las 10 iteraciones
    df_agg = df.groupby("Pregunta")[METRICAS + ["Tiempo RAG (s)"]].median()
    datos[config_name] = df_agg
    print(f"  [OK] {config_name}: {len(df)} filas -> {len(df_agg)} preguntas")

if len(datos) < 2:
    print("\nERROR: Se necesitan al menos 2 configuraciones.")
    exit(1)

# Verificar que todas las configuraciones tienen las mismas preguntas
preguntas_ref = set(datos[BASELINE].index)
for config, df_agg in datos.items():
    preguntas_config = set(df_agg.index)
    if preguntas_config != preguntas_ref:
        diff = preguntas_ref.symmetric_difference(preguntas_config)
        print(f"\n  [!] {config}: diferencia en preguntas ({len(diff)} distintas)")

n_preguntas = len(preguntas_ref)
print(f"\nPreguntas comunes: {n_preguntas}")
print(f"Configuraciones cargadas: {len(datos)}")

# ==========================================
# ESTADISTICAS DESCRIPTIVAS
# ==========================================

desc_rows = []
for config in ARCHIVOS.values():
    if config not in datos:
        continue
    row = {"Config": config}
    for m in METRICAS:
        vals = datos[config][m].dropna()
        row[METRICAS_CORTAS[m] + " media"] = round(vals.mean(), 4)
        row[METRICAS_CORTAS[m] + " mediana"] = round(vals.median(), 4)
        row[METRICAS_CORTAS[m] + " std"] = round(vals.std(), 4)
    desc_rows.append(row)

df_desc = pd.DataFrame(desc_rows)

# ==========================================
# TEST DE NORMALIDAD (Shapiro-Wilk)
# ==========================================

normalidad_rows = []
for config in ARCHIVOS.values():
    if config not in datos:
        continue
    for m in METRICAS:
        vals = datos[config][m].dropna().values
        if len(vals) < 3:
            continue
        # estadistico de Shapiro-Wilk y p-value
        stat_sw, p_sw = stats.shapiro(vals)
        # asimetria y curtosis para diagnostico complementario
        sk = stats.skew(vals)
        kt = stats.kurtosis(vals)
        
        normalidad_rows.append({
            "Config": config,
            "Metrica": METRICAS_CORTAS[m],
            "n": len(vals),
            "W stat": round(stat_sw, 4),
            "p-value": round(p_sw, 6),
            "Skewness": round(sk, 3),
            "Kurtosis": round(kt, 3),
            "Normal (p>=0.05)": "si" if p_sw >= 0.05 else "no",
        })

df_norm = pd.DataFrame(normalidad_rows)

# ==========================================
# TEST DE FRIEDMAN (por grupo de experimentos)
# ==========================================

friedman_results = []

for grupo_name, configs in GRUPOS.items():
    configs_disponibles = [c for c in configs if c in datos]
    if len(configs_disponibles) < 3:
        print(f"\n  {grupo_name}: insuficientes configuraciones ({len(configs_disponibles)}/3)")
        continue


    # Alinear por pregunta
    preguntas_comunes = set(datos[configs_disponibles[0]].index)
    for c in configs_disponibles[1:]:
        preguntas_comunes &= set(datos[c].index)
    preguntas_comunes = sorted(preguntas_comunes)

    for metrica in METRICAS:
        # Construir matriz y eliminar filas con NaN
        matrix = np.column_stack([
            datos[c].loc[preguntas_comunes, metrica].values
            for c in configs_disponibles
        ])
        mask_valid = ~np.isnan(matrix).any(axis=1)
        matrix_clean = matrix[mask_valid]
        n_valid = len(matrix_clean)

        if n_valid < 10:
            continue

        arrays = [matrix_clean[:, i] for i in range(matrix_clean.shape[1])]

        try:
            stat_f, p_f = stats.friedmanchisquare(*arrays)
            sig = significance_stars(p_f)
            friedman_results.append({
                "Grupo": grupo_name,
                "Metrica": METRICAS_CORTAS[metrica],
                "Friedman chi2": round(stat_f, 4),
                "p-value": round(p_f, 6),
                "Significancia": sig,
                "n": n_valid,
            })
        except Exception as e:
            print(f"  {METRICAS_CORTAS[metrica]:15s}: ERROR - {e}")

# ==========================================
# WILCOXON SIGNED-RANK vs BASELINE
# ==========================================

wilcoxon_raw = []

if BASELINE not in datos:
    print(f"ERROR: Baseline '{BASELINE}' no encontrado")
    exit(1)

for config in ARCHIVOS.values():
    if config == BASELINE or config not in datos:
        continue

    for metrica in METRICAS:
        resultado = pair_stats(config, BASELINE, metrica, datos)
        if not resultado:
            continue
            
        x_clean, y_clean, stat_w, p_w, r = resultado
        
        efecto = interpret_effect(r)
        media_diff = np.mean(x_clean) - np.mean(y_clean)

        wilcoxon_raw.append({
            "Config": config,
            "Metrica": METRICAS_CORTAS[metrica],
            "Mediana config": round(np.median(x_clean), 4),
            "Mediana baseline": round(np.median(y_clean), 4),
            "Diff medias": round(media_diff, 4),
            "W stat": round(stat_w, 2),
            "p-value": p_w,
            "r (effect size)": round(r, 4),
            "Efecto": efecto,
            "n": len(x_clean),
        })

# ==========================================
# CORRECCION DE HOLM (por grupo de experimento)
# ==========================================
# Mapeo para aplicar Holm dentro de cada grupo
CONFIG_A_GRUPO = {}
for grupo_name, configs in GRUPOS.items():
    for c in configs:
        if c != BASELINE:
            CONFIG_A_GRUPO[c] = grupo_name
# Configs que no estan en ningun grupo de 3
CONFIG_A_GRUPO.setdefault("MQ-B4x3", "Multi Query")
CONFIG_A_GRUPO.setdefault("MQ-B3x4", "Multi Query")
CONFIG_A_GRUPO.setdefault("META-B", "Metadata")

wilcoxon_final = []

for metrica_corta in METRICAS_CORTAS.values():
    subset = [r for r in wilcoxon_raw if r["Metrica"] == metrica_corta]
    if not subset:
        continue

    # Agrupar por grupo de experimento
    grupos_dict = {}
    for r in subset:
        grupo = CONFIG_A_GRUPO.get(r["Config"], "Otro")
        grupos_dict.setdefault(grupo, []).append(r)

    # Aplicar Holm dentro de cada grupo
    for grupo, items in grupos_dict.items():
        if len(items) == 1:
            # Un solo config en el grupo: sin correccion
            r = items[0]
            r["Grupo"] = grupo
            r["p-value (Holm)"] = round(r["p-value"], 6)
            r["Significancia"] = significance_stars(r["p-value"])
            r["p-value"] = round(r["p-value"], 6)
            wilcoxon_final.append(r)
        else:
            pvals_for_holm = [(r["Config"], r["p-value"]) for r in items]
            corrected = holm_correction(pvals_for_holm)
            corr_dict = {name: p_corr for name, _, p_corr in corrected}
            for r in items:
                r["Grupo"] = grupo
                r["p-value (Holm)"] = round(corr_dict[r["Config"]], 6)
                r["Significancia"] = significance_stars(r["p-value (Holm)"])
                r["p-value"] = round(r["p-value"], 6)
                wilcoxon_final.append(r)

# Mostrar resultados
df_wilcoxon = pd.DataFrame(wilcoxon_final)

for metrica_corta in METRICAS_CORTAS.values():
    subset = df_wilcoxon[df_wilcoxon["Metrica"] == metrica_corta]
    if subset.empty:
        continue

# ==========================================
# WILCOXON INTRAGRUPO
# ==========================================

intra_results = []

for grupo_name, configs in GRUPOS.items():
    configs_disponibles = [c for c in configs if c in datos]
    if len(configs_disponibles) < 2:
        continue

    pares = []
    for i in range(len(configs_disponibles)):
        for j in range(i + 1, len(configs_disponibles)):
            pares.append((configs_disponibles[i], configs_disponibles[j]))

    for metrica in METRICAS:
        pvals_for_holm = []
        results_temp = []

        for c1, c2 in pares:
            resultado = pair_stats(c1, c2, metrica, datos)
            if not resultado:
                continue
                
            x_clean, y_clean, stat_w, p_w, r = resultado

            pvals_for_holm.append((f"{c1} vs {c2}", p_w))
            results_temp.append({
                "Grupo": grupo_name,
                "Par": f"{c1} vs {c2}",
                "Metrica": METRICAS_CORTAS[metrica],
                "Med. A": round(np.median(x_clean), 4),
                "Med. B": round(np.median(y_clean), 4),
                "p-value": p_w,
                "r": round(r, 4),
                "Efecto": interpret_effect(r),
            })

        # Correccion Holm dentro del grupo
        if pvals_for_holm:
            corrected = holm_correction(pvals_for_holm)
            corr_dict = {name: p_corr for name, _, p_corr in corrected}

            for res in results_temp:
                res["p-value (Holm)"] = round(corr_dict[res["Par"]], 6)
                res["Significancia"] = significance_stars(res["p-value (Holm)"])
                res["p-value"] = round(res["p-value"], 6)
                intra_results.append(res)


# MQ: solo 2 configs, comparacion directa
mq_configs = ["MQ-B4x3", "MQ-B3x4"]
mq_disponibles = [c for c in mq_configs if c in datos]
if len(mq_disponibles) == 2:
    mq_rows = []
    c1, c2 = mq_disponibles

    for metrica in METRICAS:
        resultado = pair_stats(c1, c2, metrica, datos)
        if not resultado:
            continue
            
        x_clean, y_clean, stat_w, p_w, r = resultado
        
        mq_rows.append({
            "Grupo": "Multi Query",
            "Par": f"{c1} vs {c2}",
            "Metrica": METRICAS_CORTAS[metrica],
            "Med. A": round(np.median(x_clean), 4),
            "Med. B": round(np.median(y_clean), 4),
            "p-value": round(p_w, 6),
            "p-value (Holm)": round(p_w, 6),
            "Significancia": significance_stars(p_w),
            "r": round(r, 4),
            "Efecto": interpret_effect(r),
        })
        
    if mq_rows:
        intra_results.extend(mq_rows)

# ==========================================
# EXPORTACION A EXCEL
# ==========================================
ARCHIVO_SALIDA = "analisis_estadistico.xlsx"

with pd.ExcelWriter(ARCHIVO_SALIDA, engine="openpyxl") as writer:
    # Hoja 1: Descriptivas
    pd.DataFrame(desc_rows).to_excel(writer, sheet_name="Descriptivas", index=False)
    
    # Hoja 2: Normalidad
    if normalidad_rows:
        pd.DataFrame(normalidad_rows).to_excel(writer, sheet_name="Normalidad", index=False)

    # Hoja 3: Friedman
    if friedman_results:
        pd.DataFrame(friedman_results).to_excel(writer, sheet_name="Friedman", index=False)

    # Hoja 4: Wilcoxon vs Baseline
    if wilcoxon_final:
        pd.DataFrame(wilcoxon_final).to_excel(writer, sheet_name="Wilcoxon vs Baseline", index=False)

    # Hoja 5: Wilcoxon intragrupo
    if intra_results:
        pd.DataFrame(intra_results).to_excel(writer, sheet_name="Wilcoxon Intragrupo", index=False)

print(f"\nResultados guardados en: {ARCHIVO_SALIDA}")