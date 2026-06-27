import os
import re
import pandas as pd
from collections import Counter
from langchain_community.document_loaders import PyPDFLoader


CARPETA_PDFS = "pdfs_descargados"
ARCHIVO_SALIDA = "diagnostico_metadatos_v3.xlsx"
LONGITUD_MIN_TEXTO = 500  # umbral para detectar PDFs sin texto extraible

# FUNCIONES DE EXTRACCIÓN

def extraer_provincia(texto):
    """conteo para devolver la provincia más mencionada"""
    provincias = ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva",
                  "Jaén", "Málaga", "Sevilla"]
    contador = Counter()
    for prov in provincias:
        # buscamos la version con acento y sin acento
        sin_acento = (prov.replace("á", "a").replace("é", "e")
                          .replace("í", "i").replace("ó", "o")
                          .replace("ú", "u"))
        n = len(re.findall(rf'\b{prov}\b', texto, re.IGNORECASE))
        if sin_acento != prov:
            n += len(re.findall(rf'\b{sin_acento}\b', texto, re.IGNORECASE))
        if n > 0:
            contador[prov] = n
    if not contador:
        return None
    return contador.most_common(1)[0][0]


def extraer_dificultad(texto):
    # cualquier forma del encabezado + valor de la lista cerrada
    valor_es = r'(Muy\s+Alta|Media\s*-\s*Baja|Media\s*-\s*Alta|Baja|Media|Alta)'
    valor_en = r'(Very\s+Difficult|Very\s+Hard|Difficult|Hard|Average|Medium|Moderate|Easy|Low)'

    # encabezados posibles
    patrones = [
        rf'[Dd]ificultad\W+{valor_es}',
        rf'DIFICULTAD\W+{valor_es}',
        rf'[Dd]ifficulty\W+{valor_en}',
        rf'DIFFICULTY\W+{valor_en}',
    ]
    for patron in patrones:
        m = re.search(patron, texto, re.IGNORECASE)
        if m:
            valor = m.group(1).strip().lower()
            # normalizamos ingles a espanol
            mapa = {
                "easy": "baja", "low": "baja",
                "average": "media", "medium": "media", "moderate": "media",
                "difficult": "alta", "hard": "alta",
                "very difficult": "muy alta", "very hard": "muy alta",
            }
            valor = mapa.get(valor, valor)
            # normalizamos espacios en "media - baja" etc
            valor = re.sub(r'\s*-\s*', '-', valor)
            return valor
    return None


def extraer_tipo_trayecto(texto):
    valor_es = r'(Circular|Lineal|Ida\s+y\s+vuelta)'
    valor_en = r'(Circular|Linear|Round\s+trip|Return)'

    patrones = [
        rf'TRAYECTO\W+{valor_es}',
        rf'[Tt]rayecto\W+{valor_es}',
        rf'ROUTE\W+{valor_en}',
        rf'[Rr]oute\W+{valor_en}',
    ]
    for patron in patrones:
        m = re.search(patron, texto, re.IGNORECASE)
        if m:
            valor = m.group(1).strip().lower()
            mapa = {
                "linear": "lineal",
                "round trip": "ida y vuelta",
                "return": "ida y vuelta",
            }
            valor = mapa.get(valor, valor)
            return valor
    return None


def extraer_autorizacion(texto):
    """ busca sí/no en un bloque de 80 caracteres tras el encabezado"""
    patrones_bloque = [
        r'AUTORIZACI[ÓO]N\s+ESPECIAL\W+([^\n]{2,80})',
        r'[Aa]utorizaci[óo]n\s+especial\W+([^\n]{2,80})',
        r'SPECIAL\s+AUTHORIS[AZ]TION\W+([^\n]{2,80})',
        r'[Ss]pecial\s+authoris[az]tion\W+([^\n]{2,80})',
    ]
    for patron in patrones_bloque:
        m = re.search(patron, texto, re.IGNORECASE)
        if m:
            valor = m.group(1).strip().lower()
            # detectamos negacion explicita
            if re.search(r'no\s+es\s+necesaria|not\s+required|no\b', valor):
                return False
            # afirmacion explicita
            if re.search(r'\bnecesaria\b|\brequired\b|\bs[íi]\b|\byes\b', valor):
                return True
    return None


def extraer_municipios(texto):
    patrones = [
        # PROVINCIA / MUNICIPIOS\nCadiz / Tarifa
        r'PROVINCIA\s*/\s*MUNICIPIOS?\W+([^\n]{4,150})',
        r'[Pp]rovincia\s*/\s*[Mm]unicipios?\W+([^\n]{4,150})',
        # version inglesa
        r'PROVINCE\s*/\s*MUNICIPALIT(?:Y|IES)\W+([^\n]{4,150})',
        r'[Pp]rovince\s*/\s*[Mm]unicipalit(?:y|ies)\W+([^\n]{4,150})',
        # variante con "Términos municipales"
        r'T[ée]rminos?\s+municipales?[^:]*:\s*([^\n]{4,150})',
    ]
    for patron in patrones:
        m = re.search(patron, texto)
        if m:
            valor = m.group(1).strip()
            # quitamos el prefijo de provincia si esta presente
            if '/' in valor:
                partes = [p.strip() for p in valor.split('/', 1)]
                if len(partes) == 2:
                    valor = partes[1]
            return valor
    return None



def extraer_metadatos(texto, nombre_archivo):
    return {
        "source": nombre_archivo,
        "provincia": extraer_provincia(texto),
        "municipios": extraer_municipios(texto),
        "dificultad": extraer_dificultad(texto),
        "tipo_trayecto": extraer_tipo_trayecto(texto),
        "autorizacion": extraer_autorizacion(texto),
    }


if not os.path.exists(CARPETA_PDFS):
    exit()

resultados, errores = [], []

for filename in sorted(f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith('.pdf')):
    try:
        docs = PyPDFLoader(os.path.join(CARPETA_PDFS, filename)).load()
        if not docs:
            errores.append({"archivo": filename, "error": "PDF vacio"})
            continue
        
        texto = docs[0].page_content
        if len(texto.strip()) < LONGITUD_MIN_TEXTO:
            meta = {
                "source": filename, "provincia": None, "municipios": None,
                "dificultad": None, "tipo_trayecto": None, "autorizacion": None,
            }
            meta["texto_primera_pagina"] = texto
            meta["num_paginas"] = len(docs)
            meta["texto_corto"] = True
            resultados.append(meta)
            continue

        meta = extraer_metadatos(texto, filename)
        meta["texto_primera_pagina"] = texto
        meta["num_paginas"] = len(docs)
        meta["texto_corto"] = False
        resultados.append(meta)
    except Exception as e:
        errores.append({"archivo": filename, "error": str(e)})

if resultados:
    pd.DataFrame(resultados).to_excel(ARCHIVO_SALIDA, index=False)

if errores:
    pd.DataFrame(errores).to_excel("diagnostico_errores_v3.xlsx", index=False)
