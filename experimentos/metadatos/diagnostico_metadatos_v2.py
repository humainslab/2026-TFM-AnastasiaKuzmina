import os
import re
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader


CARPETA_PDFS = "pdfs_descargados"
ARCHIVO_SALIDA = "diagnostico_metadatos_v2.xlsx"

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
    # formato: X hour(s) and Y minute(s)
    m = re.search(r'(\d+)\s*hour[s]?\s*(?:and\s*)?(\d+)?\s*minute[s]?', texto, re.IGNORECASE)
    if m:
        horas = int(m.group(1))
        minutos = int(m.group(2)) if m.group(2) else 0
        return round(horas + minutos / 60, 2)
    # formato: X h Y m
    m = re.search(r'(\d+)\s*h\s*(\d+)?\s*m\b', texto, re.IGNORECASE)
    if m:
        horas = int(m.group(1))
        minutos = int(m.group(2)) if m.group(2) else 0
        return round(horas + minutos / 60, 2)
    # formato: solo minutos
    m = re.search(r'(\d+)\s*minuto[s]?', texto, re.IGNORECASE)
    if m:
        return round(int(m.group(1)) / 60, 2)
    m = re.search(r'(\d+)\s*minute[s]?', texto, re.IGNORECASE)
    if m:
        return round(int(m.group(1)) / 60, 2)
    # formato: solo horas X h
    m = re.search(r'(\d+)\s*h(?:oras?|ours?)?\b', texto, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def extraer_metadatos_pdf(texto_completo, nombre_archivo):
    meta = {
        "source":           nombre_archivo,
        "tipo_documento":   None,
        "idioma":           None,
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

    # deteccion de idioma
    es_ingles = bool(re.search(r'\b(LENGTH|DIFFICULTY|ESTIMATED\s+TIME|PROVINCE)\b', texto_completo))
    meta["idioma"] = "en" if es_ingles else "es"

    # tipo de documento 
    if re.search(r'ETAPA\s+\d+|STAGE\s+\d+', texto_completo, re.IGNORECASE):
        meta["tipo_documento"] = "guia_etapa"
    elif re.search(r'TRAYECTO|LONGITUD|DIFICULTAD|LENGTH|DIFFICULTY|ROUTE\b', texto_completo, re.IGNORECASE):
        meta["tipo_documento"] = "ficha_sendero"
    else:
        meta["tipo_documento"] = "otro"

    # nombre del sendero
    candidatos = []

    # nombre va despues de la palabra clave
    patrones_despues = [
        r'(?:^|\n)\s*endero\s*\n\s*([A-ZÁÉÍÓÚÜÑ][^\n]{3,60})',
        r'(?:^|\n)\s*\brail\s*\n\s*([A-Z][^\n]{3,60})',
        r'(?:^|\n)\s*[Ss]endero\s*\n\s*([A-ZÁÉÍÓÚÜÑ][^\n]{3,60})',
        r'(?:^|\n)\s*[Tt]rail\s*\n\s*([A-Z][^\n]{3,60})',
    ]
    for patron in patrones_despues:
        for m in re.finditer(patron, texto_completo):
            candidatos.append(m.group(1).strip())

    # nombre va ANTES de la palabra clave
    patrones_antes = [
        r'([A-ZÁÉÍÓÚÜÑ][^\n]{3,60})\n(?:[A-ZÁÉÍÓÚÜÑ\s]+\n)?endero',
        r'([A-Z][^\n]{3,60})\n(?:[A-Z\s]+\n){0,2}[Tt]rail\b',
    ]
    for patron in patrones_antes:
        for m in re.finditer(patron, texto_completo):
            candidatos.append(m.group(1).strip())

    # guias de etapa
    for patron in [r'ETAPA\s+\d+\s*[:\-]?\s*([^\n]{5,80})',
                   r'STAGE\s+\d+\s*[:\-]?\s*([^\n]{5,80})']:
        for m in re.finditer(patron, texto_completo):
            candidatos.append(m.group(1).strip())

    # limpiamos candidatos malos
    candidatos_limpios = []
    for c in candidatos:
        if len(c) < 4:
            continue
        if not re.match(r'^[A-ZÁÉÍÓÚÜÑ]', c):
            continue
        # descartamos encabezados genericos
        generic = ["MÁS INFORMACIÓN", "MAS INFORMACION", "MORE INFORMATION",
                   "BUENAS PRÁCTICAS", "GOOD PRACTICE", "TELÉFONO",
                   "EMERGENCIA", "CONSEJERÍA"]
        if any(g in c.upper() for g in generic):
            continue
        candidatos_limpios.append(c)

    # elegimos el candidato que aparece mas veces
    if candidatos_limpios:
        stopwords = {"de", "del", "la", "el", "los", "las", "y", "en",
                     "the", "of", "and", "trail"}
        def puntuacion(candidato):
            palabras = [p for p in re.findall(r'\w+', candidato.lower())
                        if len(p) > 3 and p not in stopwords]
            if not palabras:
                return 0
            return sum(len(re.findall(re.escape(p), texto_completo.lower()))
                       for p in palabras)

        mejor = max(candidatos_limpios, key=puntuacion)
        meta["nombre_sendero"] = mejor

    # parque 
    candidatos_parque = []
    patrones_parque = [
        r'Parque\s+(?:Natural|Nacional|Regional)\s+(?:de\s+)?([A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s]{3,50}?)(?:\.|\n|,)',
        r'PARQUE\s+NATURAL\s*\n\s*([A-ZÁÉÍÓÚÜÑ][^\n]{3,60})',
        r'Nature\s+Park\s+(?:of\s+)?([A-Z][^\n\.]{3,60})',
    ]
    for patron in patrones_parque:
        for m in re.finditer(patron, texto_completo):
            candidatos_parque.append(m.group(1).strip())

    # filtramos candidatos corruptos
    malos = ["sendero", "está", "esta ", "represen", "camino", "otros",
             "el ", "los ", "las "]
    candidatos_parque_limpios = []
    for c in candidatos_parque:
        # rechazamos cortes de palabra
        if c.endswith('-') or c.endswith(','):
            continue
        if any(bad in c.lower()[:10] for bad in malos):
            continue
        if len(c) < 4:
            continue
        candidatos_parque_limpios.append(c)

    if candidatos_parque_limpios:
        from collections import Counter
        contador = Counter(candidatos_parque_limpios)
        meta["parque"] = contador.most_common(1)[0][0]

    # provincia 
    provincias = ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva",
                  "Jaén", "Málaga", "Sevilla"]
    for prov in provincias:
        if re.search(prov, texto_completo, re.IGNORECASE):
            meta["provincia"] = prov
            break

    # municipios 
    patrones_muni = [
        r'(?:PROVINCIA\s*/\s*MUNICIPIOS?)\s*\n?\s*([^\n]{4,120})',
        r'(?:PROVINCE\s*/\s*MUNICIPALIT(?:Y|IES))\s*\n?\s*([^\n]{4,120})',
        r'Términos?\s+municipales?[^:]*:\s*\n?\s*([^\n]{4,120})',
    ]
    for patron in patrones_muni:
        m = re.search(patron, texto_completo, re.IGNORECASE)
        if m:
            meta["municipios"] = m.group(1).strip()
            break

    # dificultad 
    dif = _extraer_texto(
        texto_completo,
        r'[Dd]ificultad\s*[:\n]?\s*(Baja|Media|Alta|Muy\s+Alta)'
    )
    # ingles
    if not dif:
        dif_en = _extraer_texto(
            texto_completo,
            r'[Dd]ifficulty\s*[:\n]?\s*(Easy|Average|Medium|Difficult|Hard|Very\s+Difficult|Very\s+Hard)'
        )
        if dif_en:
            # normalizamos al espanol para que sean comparables
            mapa = {
                "easy": "baja",
                "average": "media", "medium": "media",
                "difficult": "alta", "hard": "alta",
                "very difficult": "muy alta", "very hard": "muy alta",
            }
            dif = mapa.get(dif_en.lower(), dif_en)
    if dif:
        meta["dificultad"] = dif.strip().lower()

    # longitud 
    meta["longitud_km"] = _extraer_numero(
        texto_completo,
        r'(?:LONGITUD|[Dd]istancia\s+total[^:]*)\s*[:\n]?\s*([\d,\.]+)\s*km'
    )
    # ingles
    if meta["longitud_km"] is None:
        meta["longitud_km"] = _extraer_numero(
            texto_completo,
            r'LENGTH\s*[:\n]?\s*([\d,\.]+)\s*km'
        )
    # metros
    if meta["longitud_km"] is None:
        metros = _extraer_numero(
            texto_completo,
            r'[Dd]istancia\s+total[^:]*:\s*([\d\.]+)\s*(?:metros|m\b)'
        )
        if metros:
            meta["longitud_km"] = round(metros / 1000, 2)

    # tiempo estimado 
    patrones_tiempo = [
        r'TIEMPO\s+ESTIMADO\s*[:\n]?\s*([^\n]{4,40})',
        r'[Tt]iempo\s+de\s+marcha\s+estimado\s*[:\n]?\s*([^\n]{4,40})',
        r'ESTIMATED\s+TIME\s*[:\n]?\s*([^\n]{4,40})',
    ]
    for patron in patrones_tiempo:
        m = re.search(patron, texto_completo, re.IGNORECASE)
        if m:
            meta["tiempo_horas"] = _convertir_tiempo_a_horas(m.group(1))
            if meta["tiempo_horas"] is not None:
                break

    # desnivel máximo 
    patrones_desnivel = [
        r'[Dd]esnivel\s+m[áa]ximo\s*[:\n]?\s*([\d,\.]+)\s*m',
        r'MAXIMUM\s+GRADIENT\s*[:\n]?\s*([\d,\.]+)\s*m',
    ]
    for patron in patrones_desnivel:
        val = _extraer_numero(texto_completo, patron)
        if val is not None:
            meta["desnivel_max_m"] = val
            break

    # tipo de trayecto 
    trayecto = _extraer_texto(
        texto_completo,
        r'TRAYECTO\s*\n?\s*(Circular|Lineal|Ida\s+y\s+vuelta)'
    )
    if not trayecto:
        # ingles
        tr_en = _extraer_texto(
            texto_completo,
            r'ROUTE\s*\n?\s*(Circular|Linear|Round\s+trip|Return)'
        )
        if tr_en:
            mapa = {"linear": "lineal", "round trip": "ida y vuelta", "return": "ida y vuelta"}
            trayecto = mapa.get(tr_en.lower(), tr_en)
    if not trayecto and meta["tipo_documento"] == "guia_etapa":
        trayecto = "lineal"
    if trayecto:
        meta["tipo_trayecto"] = trayecto.strip().lower()

    #autorización especial 
    patrones_auth = [
        r'AUTORIZACIÓN\s+ESPECIAL\s*\n?\s*(No\s+es\s+necesaria|Sí|Si|Necesaria)',
        r'SPECIAL\s+AUTHORIS[AZ]TION\s*\n?\s*(Not\s+required|Required|Yes|No)',
    ]
    for patron in patrones_auth:
        m = re.search(patron, texto_completo, re.IGNORECASE)
        if m:
            valor = m.group(1).strip().lower()
            meta["autorizacion"] = not ("no" in valor or "not required" in valor)
            break

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
    pd.DataFrame(errores).to_excel("diagnostico_errores_v2.xlsx", index=False)

