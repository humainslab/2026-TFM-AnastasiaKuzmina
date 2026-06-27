import os
import re
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader


CARPETA_PDFS = "pdfs_descargados"
ARCHIVO_SALIDA = "diagnostico_metadatos_v1.xlsx"

# FUNCIONES DE EXTRACCIÓN

def _extraer_numero(texto, patron):
    m = re.search(patron, texto, re.IGNORECASE)
    if not m:
        return None
    valor_str = m.group(1).replace(',', '.').replace(' ', '')
    try:
        return float(valor_str)
    except ValueError:
        return None

def _extraer_texto(texto, patron):
    m = re.search(patron, texto, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _convertir_tiempo_a_horas(texto):
    # formato: X hora(s) y Y minuto(s)
    m = re.search(r'(\d+)\s*hora[s]?\s*(?:y\s*)?(\d+)?\s*minuto[s]?', texto, re.IGNORECASE)
    if m:
        horas = int(m.group(1))
        minutos = int(m.group(2)) if m.group(2) else 0
        return round(horas + minutos / 60, 2)
    # formato: X h Y m
    m = re.search(r'(\d+)\s*h\s*(\d+)?\s*m', texto, re.IGNORECASE)
    if m:
        horas = int(m.group(1))
        minutos = int(m.group(2)) if m.group(2) else 0
        return round(horas + minutos / 60, 2)
    # formato: solo horas X h
    m = re.search(r'(\d+)\s*h(?:oras?)?', texto, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def extraer_metadatos_pdf(texto_completo, nombre_archivo):
    meta = {
        "source":           nombre_archivo,
        "tipo_documento":   None,
        "nombre_sendero":   None,
        "parque":           None,
        "provincia":        None,
        "municipios":       None,
        "dificultad":       None,
        "longitud_km":      None,
        "tiempo_horas":     None,
        "desnivel_max_m":   None,
        "tipo_trayecto":    None,
        "autorizacion":     None,
    }

    # tipo de documento 
    if re.search(r'ETAPA\s+\d+', texto_completo, re.IGNORECASE):
        meta["tipo_documento"] = "guia_etapa"
    elif re.search(r'TRAYECTO|LONGITUD|DIFICULTAD', texto_completo, re.IGNORECASE):
        meta["tipo_documento"] = "ficha_sendero"
    else:
        meta["tipo_documento"] = "otro"

    # nombre del sendero
    nombre = _extraer_texto(texto_completo, r'[Ss]endero\s*\n?\s*([A-ZÁÉÍÓÚÜÑ][^\n]{3,60})')
    if not nombre:
        nombre = _extraer_texto(texto_completo, r'ETAPA\s+\d+\s+([^\n]{5,80})')
    meta["nombre_sendero"] = nombre

    # parque 
    meta["parque"] = _extraer_texto(
        texto_completo,
        r'(?:Parque\s+(?:Natural|Nacional|Regional))\s+(?:de\s+)?([^\n\.]{4,60})'
    )

    # provincia 
    provincias = ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva",
                  "Jaén", "Málaga", "Sevilla"]
    for prov in provincias:
        if re.search(prov, texto_completo, re.IGNORECASE):
            meta["provincia"] = prov
            break

    # municipios 
    meta["municipios"] = _extraer_texto(
        texto_completo,
        r'(?:PROVINCIA\s*/\s*MUNICIPIOS?|Términos?\s+municipales?[^:]*:)\s*\n?\s*([^\n]{4,120})'
    )

    # dificultad 
    dif = _extraer_texto(texto_completo, r'DIFICULTAD\s*\n?\s*(Baja|Media|Alta|Muy\s+Alta)')
    if not dif:
        dif = _extraer_texto(texto_completo, r'[Dd]ificultad\s*:?\s*(Baja|Media|Alta|Muy\s+Alta)')
    if dif:
        meta["dificultad"] = dif.strip().lower()

    # longitud 
    meta["longitud_km"] = _extraer_numero(
        texto_completo,
        r'(?:LONGITUD|[Dd]istancia\s+total[^:]*)\s*[:\n]?\s*([\d,\.]+)\s*km'
    )
    if meta["longitud_km"] is None:
        metros = _extraer_numero(
            texto_completo,
            r'[Dd]istancia\s+total[^:]*:\s*([\d\.]+)\s*(?:metros|m\b)'
        )
        if metros:
            meta["longitud_km"] = round(metros / 1000, 2)

    # tiempo estimado 
    m_tiempo = re.search(
        r'(?:TIEMPO\s+ESTIMADO|[Tt]iempo\s+de\s+marcha\s+estimado)\s*[:\n]?\s*([^\n]{4,30})',
        texto_completo
    )
    if m_tiempo:
        meta["tiempo_horas"] = _convertir_tiempo_a_horas(m_tiempo.group(1))

    # desnivel máximo 
    meta["desnivel_max_m"] = _extraer_numero(
        texto_completo,
        r'[Dd]esnivel\s+m[áa]ximo\s*[:\n]?\s*([\d,\.]+)\s*m'
    )

    # tipo de trayecto 
    trayecto = _extraer_texto(texto_completo, r'TRAYECTO\s*\n?\s*(Circular|Lineal|Ida\s+y\s+vuelta)')
    if not trayecto and meta["tipo_documento"] == "guia_etapa":
        trayecto = "lineal"
    if trayecto:
        meta["tipo_trayecto"] = trayecto.strip().lower()

    # autorización especial
    m_auth = re.search(
        r'AUTORIZACIÓN\s+ESPECIAL\s*\n?\s*(No\s+es\s+necesaria|Sí|Si|Necesaria)',
        texto_completo, re.IGNORECASE
    )
    if m_auth:
        valor = m_auth.group(1).strip().lower()
        meta["autorizacion"] = False if "no" in valor else True

    return meta

if not os.path.exists(CARPETA_PDFS):
    exit()

resultados, errores = [], []

for filename in [f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith('.pdf')]:
    try:
        docs = PyPDFLoader(os.path.join(CARPETA_PDFS, filename)).load()
        meta = extraer_metadatos_pdf(" ".join(d.page_content for d in docs), filename)
        meta["num_paginas"] = len(docs)
        resultados.append(meta)
    except Exception as e:
        errores.append({"archivo": filename, "error": str(e)})

if resultados:
    pd.DataFrame(resultados).to_excel(ARCHIVO_SALIDA, index=False)

if errores:
    pd.DataFrame(errores).to_excel("diagnostico_errores.xlsx", index=False)
