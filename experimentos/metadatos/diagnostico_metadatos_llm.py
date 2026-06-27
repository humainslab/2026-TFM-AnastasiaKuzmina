import os
import json
import re
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama


CARPETA_PDFS = "pdfs_descargados"
ARCHIVO_SALIDA = "diagnostico_metadatos_llm.xlsx"

#TOKEN
TOKEN = ''
ollama_server = os.environ.get("OLLAMA_SERVER", "")
headers = {"authorization": f"Bearer {TOKEN}"}

llm = ChatOllama(
    model="llama3.1:8b",
    base_url=ollama_server,
    client_kwargs={"headers": headers},
    temperature=0,
)

PROMPT_EXTRACCION = """Eres un extractor de metadatos. Tu tarea es leer el texto de una ficha de sendero y devolver los datos estructurados en formato JSON.

REGLAS ESTRICTAS:
- Devuelve SOLO un objeto JSON valido, sin texto adicional antes ni despues.
- Si un campo no aparece en el texto, devuelve null. NO inventes valores.
- Usa exactamente los nombres de campo indicados.
- Los valores deben ser literales del texto o su traduccion obvia al espanol.

CAMPOS A EXTRAER:
- tipo_documento: "ficha_sendero" si es una ficha de un sendero con datos tecnicos, "guia_etapa" si es una etapa de un gran recorrido, "otro" si no es ninguna de las dos cosas
- nombre_sendero: nombre propio del sendero (sin la palabra "sendero")
- parque: nombre del parque natural, nacional o regional (sin "Parque Natural de")
- provincia: una de: Almeria, Cadiz, Cordoba, Granada, Huelva, Jaen, Malaga, Sevilla
- municipios: municipios por los que pasa el sendero
- dificultad: "baja", "media", "alta" o "muy alta" (normaliza easy->baja, average->media, difficult->alta)
- longitud_km: numero en kilometros (convierte metros a km si hace falta)
- tiempo_horas: duracion estimada en horas (convierte minutos a fraccion de hora)
- desnivel_max_m: desnivel maximo en metros
- tipo_trayecto: "circular", "lineal" o "ida y vuelta"
- autorizacion: true si se necesita autorizacion especial, false si no

TEXTO DE LA FICHA:
{texto}

JSON:"""


# FUNCIONES DE EXTRACCIÓN

def parsear_json_llm(respuesta):
    """Extrae el primer bloque JSON valido de la respuesta del LLM."""
    # intentamos parsear la respuesta entera primero
    try:
        return json.loads(respuesta.strip())
    except json.JSONDecodeError:
        pass

    # buscamos un bloque {...} dentro del texto
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', respuesta, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # si sigue fallando, quitamos fences de codigo tipo ```json ... ```
    limpio = re.sub(r'^```(?:json)?\s*|\s*```$', '', respuesta.strip(), flags=re.MULTILINE)
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        return None


def normalizar_meta(data, nombre_archivo):
    """campos None y orden."""
    campos = ["tipo_documento", "nombre_sendero", "parque", "provincia",
              "municipios", "dificultad", "longitud_km", "tiempo_horas",
              "desnivel_max_m", "tipo_trayecto", "autorizacion"]
    meta = {"source": nombre_archivo}
    if data is None:
        for c in campos:
            meta[c] = None
        meta["_llm_fallo"] = True
        return meta

    for c in campos:
        val = data.get(c)
        # convertimos cadenas "null"/"none" a None real
        if isinstance(val, str) and val.strip().lower() in ("null", "none", ""):
            val = None
        meta[c] = val
    meta["_llm_fallo"] = False
    return meta


if not os.path.exists(CARPETA_PDFS):
    exit()

archivos_pdf = sorted(f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith('.pdf'))

# si hay un excel parcial de ejecuciones anteriores, lo retomamos
resultados = []
procesados = set()
if os.path.exists(ARCHIVO_SALIDA):
    try:
        df_previo = pd.read_excel(ARCHIVO_SALIDA)
        resultados = df_previo.to_dict("records")
        procesados = {r["source"] for r in resultados}
    except Exception:
        pass

errores = []

for i, filename in enumerate(archivos_pdf):
    if filename in procesados:
        continue

    try:
        # cargamos el PDF y tomamos solo la primera pagina
        docs = PyPDFLoader(os.path.join(CARPETA_PDFS, filename)).load()
        if not docs:
            errores.append({"archivo": filename, "error": "PDF vacio"})
            continue
        texto_primera_pagina = docs[0].page_content
        num_paginas = len(docs)

        # llamamos al LLM
        prompt = PROMPT_EXTRACCION.format(texto=texto_primera_pagina)
        respuesta = llm.invoke(prompt).content

        # parseamos la respuesta y normalizamos los campos
        data = parsear_json_llm(respuesta)
        meta = normalizar_meta(data, filename)
        meta["num_paginas"] = num_paginas
        resultados.append(meta)

        # guardamos cada 10 PDFs para no perder progreso si el script cae
        if (i + 1) % 10 == 0:
            pd.DataFrame(resultados).to_excel(ARCHIVO_SALIDA, index=False)

    except Exception as e:
        errores.append({"archivo": filename, "error": str(e)})

pd.DataFrame(resultados).to_excel(ARCHIVO_SALIDA, index=False)

if errores:
    pd.DataFrame(errores).to_excel("diagnostico_errores_llm.xlsx", index=False)
