# ==========================================
# IMPORT DE LIBRERÍAS
# ==========================================
import os
import re
import time
import pandas as pd
import requests

# para descargar el texto de las páginas web
from langchain_community.document_loaders import WebBaseLoader
# para dividir el texto chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
# base de datos vectorial
from langchain_community.vectorstores import FAISS
# para convertir texto en vectores numéricos
from langchain_ollama import OllamaEmbeddings, ChatOllama
# herramientas para construir RAG
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# memoria
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# ==========================================
# CONFIGURACIÓN GENERAL Y SERVIDOR
# ==========================================
# configuracion del test
ARCHIVO_PREGUNTAS = "Escenarios_medio.xlsx"
ARCHIVO_SALIDA = "resultados_ragas_META-B.xlsx"
ARCHIVO_ENLACES_PDF = "Enlaces PDF.xlsx"
NUM_REPETICIONES = 10

# nombre de la carpeta donde se va a guardar la base de datos
DB_NAME = "andalucia_db_metadata"

#TOKEN
TOKEN = ''
ollama_server = os.environ.get("OLLAMA_SERVER", "")
headers = {"authorization": f"Bearer {TOKEN}"}

# lista de URLs
urls = [
    "https://www.juntadeandalucia.es/medioambiente/portal/landing-page-%C3%ADndice/-/asset_publisher/zX2ouZa4r1Rf/content/oficinas-y-centros-de-visitantes-de-espacios-naturales-protegidos-de-andaluc-c3-ada/20151",
    "https://www.juntadeandalucia.es/medioambiente/portal/web/ventanadelvisitante/inicio"
]

# convierte el texto en listas de números para que el ordenador pueda buscar información 
embeddings = OllamaEmbeddings(
    model="llama3.1:8b",
    base_url=ollama_server,
    client_kwargs={"headers": headers}
)

# el modelo
llm = ChatOllama(
    model="llama3.1:8b",
    base_url=ollama_server,
    client_kwargs={"headers": headers}
)

# dividimos el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ==========================================
# EXTRACCIÓN DE METADATOS (REGEX v3)
# ==========================================
# extraemos 5 campos basicos de la primera pagina del PDF
# (provincia, municipios, dificultad, tipo_trayecto, autorizacion)
# y los inyectamos como cabecera en cada chunk antes del embedding

# umbral para descartar PDFs sin texto extraible (PDFs graficos, OCR)
LONGITUD_MIN_TEXTO = 500

# funcion para extraer la provincia: busqueda libre
def extraer_provincia(texto):
    from collections import Counter
    provincias = ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva",
                  "Jaén", "Málaga", "Sevilla"]
    contador = Counter()
    for prov in provincias:
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

# funcion para extraer dificultad: encabezado como ancla
def extraer_dificultad(texto):
    valor_es = r'(Muy\s+Alta|Media\s*-\s*Baja|Media\s*-\s*Alta|Baja|Media|Alta)'
    valor_en = r'(Very\s+Difficult|Very\s+Hard|Difficult|Hard|Average|Medium|Moderate|Easy|Low)'
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
            mapa = {
                "easy": "baja", "low": "baja",
                "average": "media", "medium": "media", "moderate": "media",
                "difficult": "alta", "hard": "alta",
                "very difficult": "muy alta", "very hard": "muy alta",
            }
            valor = mapa.get(valor, valor)
            valor = re.sub(r'\s*-\s*', '-', valor)
            return valor
    return None

# funcion para extraer tipo de trayecto: encabezado como ancla
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
            mapa = {"linear": "lineal", "round trip": "ida y vuelta", "return": "ida y vuelta"}
            valor = mapa.get(valor, valor)
            return valor
    return None

# funcion para extraer autorizacion especial: buscamos bajo el encabezado
def extraer_autorizacion(texto):
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
            if re.search(r'no\s+es\s+necesaria|not\s+required|no\b', valor):
                return False
            if re.search(r'\bnecesaria\b|\brequired\b|\bs[íi]\b|\byes\b', valor):
                return True
    return None

# funcion para extraer municipios: lista abierta, requiere ancla
def extraer_municipios(texto):
    patrones = [
        r'PROVINCIA\s*/\s*MUNICIPIOS?\W+([^\n]{4,150})',
        r'[Pp]rovincia\s*/\s*[Mm]unicipios?\W+([^\n]{4,150})',
        r'PROVINCE\s*/\s*MUNICIPALIT(?:Y|IES)\W+([^\n]{4,150})',
        r'[Pp]rovince\s*/\s*[Mm]unicipalit(?:y|ies)\W+([^\n]{4,150})',
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

# funcion principal que devuelve los 5 metadatos de la primera pagina
def extraer_metadatos_pdf(texto_primera_pagina):
    if len(texto_primera_pagina.strip()) < LONGITUD_MIN_TEXTO:
        # PDF sin texto extraible
        return {}
    return {
        "provincia":     extraer_provincia(texto_primera_pagina),
        "municipios":    extraer_municipios(texto_primera_pagina),
        "dificultad":    extraer_dificultad(texto_primera_pagina),
        "tipo_trayecto": extraer_tipo_trayecto(texto_primera_pagina),
        "autorizacion":  extraer_autorizacion(texto_primera_pagina),
    }

# funcion para construir la cabecera estructurada que se inyecta en cada chunk
def construir_cabecera(meta):
    partes = []
    if meta.get("provincia"):
        partes.append(f"Provincia: {meta['provincia']}")
    if meta.get("municipios"):
        partes.append(f"Municipios: {meta['municipios']}")
    if meta.get("dificultad"):
        partes.append(f"Dificultad: {meta['dificultad']}")
    if meta.get("tipo_trayecto"):
        partes.append(f"Trayecto: {meta['tipo_trayecto']}")
    if meta.get("autorizacion") is not None:
        valor = "necesaria" if meta["autorizacion"] else "no necesaria"
        partes.append(f"Autorización: {valor}")
    if not partes:
        return ""
    return "[" + " | ".join(partes) + "]\n\n"

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
# creamos una función para procesar y guardar el batch actual
def process_save_batch(batch_docs, vs, current_total):
    if not batch_docs:
        return vs
        
    print(f"Procesando {len(batch_docs)} documentos...", flush=True)
    t_split_start = time.time()
    splits = text_splitter.split_documents(batch_docs)
    print(f"Split: {round(time.time() - t_split_start, 2)}s", flush=True)
    
    # inyectamos la cabecera con metadatos al inicio de cada chunk de PDF
    # los chunks de URLs (que no tienen 'metadatos_estructurados') se dejan tal cual
    for split in splits:
        meta_estruct = split.metadata.get('metadatos_estructurados')
        if meta_estruct:
            cabecera = construir_cabecera(meta_estruct)
            if cabecera:
                split.page_content = cabecera + split.page_content
    
    if splits:
        print(f"Generando embeddings para {len(splits)} fragmentos...", flush=True)
        t_embed_start = time.time()
        
        # procesamos en mini-batches para mostrar progreso
        mini_batch_size = 5
        total_splits = len(splits)
        
        # inicializamos vectorstore si es necesario
        start_index = 0
        if vs is None:
            print(f"Procesando primeros {min(mini_batch_size, total_splits)}...", flush=True)
            vs = FAISS.from_documents(documents=splits[:mini_batch_size], embedding=embeddings)
            start_index = mini_batch_size
            
        # añadimos el resto
        for i in range(start_index, total_splits, mini_batch_size):
            end_idx = min(i + mini_batch_size, total_splits)
            print(f"Procesando {i+1} a {end_idx} de {total_splits}...", flush=True)
            vs.add_documents(documents=splits[i:end_idx])
            
        print(f"Embeddings: {round(time.time() - t_embed_start, 2)}s", flush=True)
        
        # guardamos en base de datos
        print(f"Guardando base de datos...", flush=True)
        t_save_start = time.time()
        vs.save_local(DB_NAME)
        t_save_end = time.time()
        print(f"Guardado completado en {round(t_save_end - t_save_start, 2)}s", flush=True)
    
    return vs

# creamos una función para descargar PDFs
def descargar_pdfs(lista_links, carpeta_destino="pdfs_descargados"):
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    print(f"\nDescargando PDFs ({len(lista_links)} encontrados)...", flush=True)
    count = 0
    total = len(lista_links)
    for i, url in enumerate(lista_links):
        url = str(url).strip()
        if not url.lower().startswith('http'): continue
        filename = url.split('/')[-1]
        if not filename.lower().endswith('.pdf'): filename += ".pdf"
        # limpiamos el nombre del archivo
        filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).strip()
        filepath = os.path.join(carpeta_destino, filename)
        
        if not os.path.exists(filepath):
            try:
                print(f"[{i+1}/{total}] Descargando: {url}")
                with open(filepath, 'wb') as f:
                    f.write(requests.get(url, timeout=30).content)
                count += 1
            except Exception as e: print(f" [!] Error: {e}")
    print(f"Descarga finalizada: {count} documentos nuevos\n")

# comprobamos si la base de datos ya existe
# creamos una lista para acumular documentos temporalmente
current_batch = []
BATCH_SIZE = 5
total_processed = 0
vectorstore = None

if os.path.exists(DB_NAME):
    print(f"\nBase de datos '{DB_NAME}' encontrada. Se cargarán y añadirán nuevos documentos", flush=True)
    vectorstore = FAISS.load_local(DB_NAME, embeddings, allow_dangerous_deserialization=True)
else:
    print("Base de datos no encontrada. Se creará una nueva", flush=True)

# cargamos el historial
HISTORY_FILE = "procesados_metadata.txt"
processed_files = set()
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            processed_files = {line.strip() for line in f if line.strip()}
        print(f"Historial cargado: {len(processed_files)} items")
    except: pass

# creamos la función para guardar historial(lo que ya tenemos guardado)
def save_history(new_items):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            for item in new_items:
                f.write(item + "\n")
        processed_files.update(new_items)
        print("Historial actualizado")
    except Exception as e:
        print(f"Error historial: {e}")

# ==========================================
# PROCESAMIENTO DE DATOS
# ==========================================
# cargamos el contenido de las URLs
print("\n" + "="*50)
print("FASE 1: CARGA DE SITIOS WEB")
print("="*50)

web_start_time = time.time()
web_docs_count = 0

# filtramos URLs ya procesadas
urls_a_procesar = [u for u in urls if u not in processed_files]

print(f"URLs totales: {len(urls)}")
print(f"Ya procesadas: {len(urls) - len(urls_a_procesar)}")

# cargamos el contenido de las URLs
if urls_a_procesar:
    try:
        print(f"Cargando {len(urls_a_procesar)} sitios web...", end=" ", flush=True)
        t_start = time.time()
        
        loader = WebBaseLoader(
            urls_a_procesar,
            header_template={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        )
        web_docs = loader.load()
        
        current_batch.extend(web_docs)
        web_docs_count = len(web_docs)
        print(f"Carga completada: {web_docs_count} documentos en {round(time.time() - t_start, 2)}s")
        
    except Exception as e:
        print(f"ERROR cargando URLs: {e}")

print(f"Sitios web procesados: {web_docs_count} documentos nuevos")

# guardamos los datos de la web en la base de datos
if current_batch:
    print("Guardando datos de la web en la base de datos...")
    vectorstore = process_save_batch(current_batch, vectorstore, total_processed)
    
    # actualizamos historial con las URLs que acabamos de procesar
    save_history(urls_a_procesar)
    current_batch = [] # limpiamos la lista para que no se mezcle con los PDFs
            
# descargamos los PDFs
print("\n" + "="*50)
print("FASE 2: DESCARGA DE PDFs")
print("="*50)
try:
    df_pdfs = pd.read_excel(ARCHIVO_ENLACES_PDF, header=None)
    lista_pdfs = df_pdfs[0].dropna().tolist()
    
    descargar_pdfs(lista_pdfs)
    
except Exception as e:
    print(f"Error al descargar PDFs: {e}")

# procesamos los PDFs locales
print("\n" + "="*50)
print("FASE 3: PROCESAMIENTO DE PDFs")
print("="*50)

carpeta_pdfs = "pdfs_descargados"
if os.path.exists(carpeta_pdfs):
    archivos_pdf = [f for f in os.listdir(carpeta_pdfs) if f.lower().endswith('.pdf')]
    
    # filtramos los que ya estan procesados
    archivos_nuevos = [f for f in archivos_pdf if f not in processed_files]
    
    print(f"Total PDFs encontrados: {len(archivos_pdf)}")
    print(f"Ya procesados: {len(archivos_pdf) - len(archivos_nuevos)}")
    print(f"Nuevos PDFs: {len(archivos_nuevos)}")
   
    # lista para identificar qué archivos están en el lote actual
    current_batch_filenames = []
    
    for i, filename in enumerate(archivos_nuevos):
        filepath = os.path.join(carpeta_pdfs, filename)
        try:
            print(f"[{i+1}/{len(archivos_nuevos)}] Leyendo: {filename} ...", end=" ", flush=True)
            t_load_start = time.time()
            
            # usamos PyPDFLoader de LangChain para cargar los PDFs
            loader_pdf = PyPDFLoader(filepath)
            docs_pdf = loader_pdf.load()
            
            # extraemos los metadatos estructurados de la primera pagina del PDF
            # y los guardamos en doc.metadata para inyectarlos en los chunks despues
            metadatos_pdf = {}
            if docs_pdf:
                metadatos_pdf = extraer_metadatos_pdf(docs_pdf[0].page_content)
            
            # establecemos el nombre del archivo y añadimos al batch
            for doc in docs_pdf:
                doc.metadata['source'] = filename 
                if metadatos_pdf:
                    doc.metadata['metadatos_estructurados'] = metadatos_pdf
                current_batch.append(doc)
            
            current_batch_filenames.append(filename)
            
            t_load_end = time.time()
            print(f"OK ({round(t_load_end - t_load_start, 2)}s).Buffer: {len(current_batch)} docs")
            
            total_processed += 1
            
            # guardamos si alcanzamos el limite (5 archivos o 200 paginas)
            if len(current_batch_filenames) >= BATCH_SIZE or len(current_batch) >= 200:
                vectorstore = process_save_batch(current_batch, vectorstore, total_processed)
                save_history(current_batch_filenames)
                current_batch, current_batch_filenames = [], []
            
        except Exception as e:
            print(f"ERROR: {e}")

if current_batch:
    vectorstore = process_save_batch(current_batch, vectorstore, total_processed)
    save_history(current_batch_filenames)

if vectorstore:
    print(f"\nProceso finalizado. Docs procesados: {total_processed}")
else:
    print("ERROR")

# ==========================================
# CONFIGURACIÓN DEL RAG
# ==========================================
# configuracion de retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# definimos el prompt para reformular la pregunta basándose en el historial
contextualize_q_system_prompt = (
    "Dada la historia del chat y la última pregunta del usuario, "
    "que podría hacer referencia al contexto en la historia del chat, "
    "formula una pregunta independiente que pueda entenderse sin la historia del chat. "
    "NO respondas a la pregunta, solo reformúlala si es necesario y devuélvela."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), # Aquí guardamos la historia
    ("human", "{input}"),
])

# creamos el History Aware Retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# prompt del sistema
system_prompt = (
    "Eres un guía experto en turismo activo y naturaleza de Andalucía. Tu objetivo es asesorar al usuario de forma fiable y profesional.\n\n"
 
    "REGLA 1 - FUENTES DE INFORMACIÓN:\n"
    "Usa EXCLUSIVAMENTE el contexto técnico proporcionado para responder sobre:\n"
    "- Longitud, desnivel, duración y dificultad de senderos concretos\n"
    "- Flora y fauna de parques o rutas específicas\n"
    "- Puntos de inicio, aparcamiento, señalización y balizamiento\n"
    "- Restricciones, permisos o normativa de espacios naturales\n"
    "Si esta información no está en el contexto, indícalo claramente: "
    "'No tengo datos técnicos sobre esta ruta en concreto, pero puedo orientarte sobre opciones similares.'\n\n"
 
    "REGLA 2 - CONOCIMIENTO GENERAL PERMITIDO:\n"
    "Puedes usar tu conocimiento general únicamente para:\n"
    "- Geografía: provincias de Andalucía, ubicación de municipios, distancias entre localidades\n"
    "- Clima y estacionalidad general de la región\n"
    "- Recomendaciones genéricas de equipamiento (agua, calzado, ropa)\n"
    "- Clasificación estándar de dificultad (qué significa fácil, medio, difícil en senderismo)\n\n"
 
    "REGLA 3 - FILTRO GEOGRÁFICO:\n"
    "Si los datos técnicos disponibles corresponden a una zona distinta a la consultada, "
    "indícalo antes de dar la información: "
    "'Los datos que tengo son de [zona], que está a aproximadamente [distancia] de donde mencionas. "
    "Te los comparto por si te son útiles.'\n\n"
 
    "REGLA 4 - DOCUMENTOS Y ARCHIVOS GPX:\n"
    "Si el usuario solicita mapas, archivos GPX, PDF de rutas u otros documentos descargables, "
    "explica que no puedes enviar archivos directamente y dirígele al portal oficial: "
    "'Para descargar el track GPX o el PDF oficial de esta ruta, puedes encontrarlo en el portal "
    "de la Junta de Andalucía: juntadeandalucia.es/medioambiente — sección Red de Senderos de Andalucía.'\n\n"
 
    "REGLA 5 - FORMATO:\n"
    "Usa **negrita** para nombres de rutas y conceptos clave. "
    "Usa listas con guiones para que la información sea escaneable. "
    "Sé conciso: no repitas información ya mencionada en el historial del chat.\n\n"
 
    "Información técnica disponible:\n"
    "{context}"
)
# creamos el prompt con el mensaje del sistema y la entrada del usuario
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# creamos la cadena de documentos que combina los textos
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# creamos la cadena final que une el buscador con el generador de respuestas
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ==========================================
# MODO TEST Y CHAT
# ==========================================
print("\n" + "="*60)
print("SELECCIONE EL MODO DE EJECUCIÓN")
print("1 - MODO TEST")
print("2 - MODO CHAT")
print("="*60)

while True:
    modo = input("\nIngrese 1 o 2: ").strip()
    if modo in ["1", "2"]:
        break
    print("Por favor, ingrese 1 o 2")

if modo == "1":
    # creamos el modo test
    print("\n" + "="*60 + "\nMODO TEST\n" + "="*60)
    try:
        df_preg = pd.read_excel(ARCHIVO_PREGUNTAS, header=None, engine='openpyxl')
        preguntas = [str(p).strip().replace('"', '') for p in df_preg[0].dropna() if len(str(p)) > 3]
        resultados = []
        
        for p in preguntas:
            for i in range(1, NUM_REPETICIONES + 1):
                print(f"{p[:30]}... ({i} iteracion)")
                t0 = time.time()
                try:
                    # en modo TEST enviamos historial vacío porque cada pregunta es independiente
                    result = rag_chain.invoke({"input": p, "chat_history": []})
                    resp = result['answer']
                    context_docs = result.get('context', [])
                    context_text = " ".join([doc.page_content for doc in context_docs])
                    input_text = p + " " + context_text
                    
                    fuentes = set([doc.metadata.get('source', 'Desconocido') for doc in context_docs])
                    fuentes_str = ", ".join(fuentes)
                    
                    dt = round(time.time() - t0, 4)
                    resultados.append({
                        "Pregunta": p, "Iteración": i, "Tiempo (s)": dt,
                        "Tokens Entrada": int(len(input_text.split())*1.3), "Tokens Salida": int(len(resp.split())*1.3), "Respuesta": resp,
                        "Fuentes": fuentes_str, "Contexto usado": context_text
                    })
                except Exception as e: print(f"Error {i}: {e}")

        if resultados:
            pd.DataFrame(resultados).to_excel(ARCHIVO_SALIDA, index=False)
            print(f"Resultados guardados en {ARCHIVO_SALIDA}")
    except Exception as e: print(f"Error test: {e}")

elif modo == "2":
    print("\nPuede realizar sus consultas sobre Andalucia. (Escriba 'exit' o 'stop' para salir)\n")

    # BUCLE DE CHAT CON MEMORIA
    # creamos la lista de historia vacía al empezar el chat
    chat_history = [] 

    while True:
        print("\n" + "="*60)
        query = input("\n INGRESE SU CONSULTA: ")
        # cuando salimos del bucle
        if query.lower() in ["exit", "stop"]:
            break
        
        print("Pensando...")
        try:
            # ejecutamos la cadena RAG con la pregunta del usuario
            response = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
            # añadimos la pregunta y la respuesta al historial
            chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=response["answer"]),
            ])
            
            # limitamos la memoria a los últimos 10 mensajes
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            
            print("\n" + "="*60)
            print("RESPUESTA GENERADA:")
            print("="*60)
            print(response['answer'])
            
            context_docs = response.get('context', [])
            if context_docs:
                print("\n" + "-"*40)
                print("Fuentes utilizadas:")
                for i, doc in enumerate(context_docs):
                    fuente = doc.metadata.get('source', 'Desconocido')
                    extracto = doc.page_content[:200].replace('\n', ' ') + "..."
                    print(f"[{i+1}] Fuentes: {fuente}")
                    print(f"    Contexto usado: {extracto}")
                    
            print("\n" + "-"*60)
            
        except Exception as e:
            print(f"[ERROR]: {e}")