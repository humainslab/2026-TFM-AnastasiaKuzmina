# ==========================================
# IMPORT DE LIBRERÍAS
# ==========================================
import os
import time
import pandas as pd
import requests

#para descargar el texto de las páginas web
from langchain_community.document_loaders import WebBaseLoader
#para dividir el texto chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
#base de datos vectorial
from langchain_community.vectorstores import FAISS
#para convertir texto en vectores numéricos
from langchain_ollama import OllamaEmbeddings, ChatOllama
#herramientas para construir RAG
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
#configuracion del test
ARCHIVO_PREGUNTAS = "Escenarios_corto.xlsx"
ARCHIVO_SALIDA = "resultados_metricas.xlsx"
ARCHIVO_ENLACES_PDF = "Enlaces PDF_mini.xlsx"
NUM_REPETICIONES = 10

#nombre de la carpeta donde se va a guardar la base de datos
DB_NAME = "andalucia_db"

#lista de URLs
urls = [
    "https://www.juntadeandalucia.es/medioambiente/portal/landing-page-%C3%ADndice/-/asset_publisher/zX2ouZa4r1Rf/content/oficinas-y-centros-de-visitantes-de-espacios-naturales-protegidos-de-andaluc-c3-ada/20151",
    "https://www.juntadeandalucia.es/medioambiente/portal/web/ventanadelvisitante/inicio"
]

#convierte el texto en listas de números para que el ordenador pueda buscar información
embeddings = OllamaEmbeddings(model="llama3:8b")
#el modelo
llm = ChatOllama(model="llama3:8b")

#dividimos el texto en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
#creamos una función para procesar y guardar el batch actual
def process_save_batch(batch_docs, vs, current_total):
    if not batch_docs:
        return vs
        
    print(f"Procesando {len(batch_docs)} documentos...", flush=True)
    t_split_start = time.time()
    splits = text_splitter.split_documents(batch_docs)
    print(f"Split: {round(time.time() - t_split_start, 2)}s", flush=True)
    
    if splits:
        print(f"Generando embeddings para {len(splits)} fragmentos...", flush=True)
        t_embed_start = time.time()
        
        #procesamos en mini-batches para mostrar progreso
        mini_batch_size = 5
        total_splits = len(splits)
        
        #inicializamos vectorstore si es necesario
        start_index = 0
        if vs is None:
            print(f"Procesando primeros {min(mini_batch_size, total_splits)}...", flush=True)
            vs = FAISS.from_documents(documents=splits[:mini_batch_size], embedding=embeddings)
            start_index = mini_batch_size
            
        #añadimos el resto
        for i in range(start_index, total_splits, mini_batch_size):
            end_idx = min(i + mini_batch_size, total_splits)
            print(f"Procesando {i+1} a {end_idx} de {total_splits}...", flush=True)
            vs.add_documents(documents=splits[i:end_idx])
            
        print(f"Embeddings: {round(time.time() - t_embed_start, 2)}s", flush=True)
        
        #guardamos en base de datos
        print(f"Guardando base de datos...", flush=True)
        t_save_start = time.time()
        vs.save_local(DB_NAME)
        t_save_end = time.time()
        print(f"Guardado completado en {round(t_save_end - t_save_start, 2)}s", flush=True)
    
    return vs

#creamos una función para descargar PDFs
def descargar_pdfs(lista_links, carpeta_destino="pdfs_descargados_mini"):
    if not os.path.exists(carpeta_destino): os.makedirs(carpeta_destino)
    print(f"\nDescargando PDFs ({len(lista_links)} encontrados)...", flush=True)
    count = 0
    total = len(lista_links)
    for i, url in enumerate(lista_links):
        url = str(url).strip()
        if not url.lower().startswith('http'): continue
        filename = url.split('/')[-1]
        if not filename.lower().endswith('.pdf'): filename += ".pdf"
        #limpiamos el nombre del archivo
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

#comprobamos si la base de datos ya existe
#creamos una lista para acumular documentos temporalmente
current_batch = []
BATCH_SIZE = 5
total_processed = 0
vectorstore = None

if os.path.exists(DB_NAME):
    print(f"\nBase de datos '{DB_NAME}' encontrada. Se cargarán y añadirán nuevos documentos", flush=True)
    vectorstore = FAISS.load_local(DB_NAME, embeddings, allow_dangerous_deserialization=True)
else:
    print("Base de datos no encontrada. Se creará una nueva", flush=True)

#creamos la función para guardar historial(lo que ya tenemos guardado)
def save_history(new_items):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            for item in new_items:
                f.write(item + "\n")
        processed_files.update(new_items)
        print("Historial actualizado")
    except Exception as e:
        print(f"Error historial: {e}")

#cargamos el historial
HISTORY_FILE = "procesados.txt"
processed_files = set()
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            processed_files = {line.strip() for line in f if line.strip()}
        print(f"Historial cargado: {len(processed_files)} items")
    except: pass

# ==========================================
# PROCESAMIENTO DE DATOS
# ==========================================
#cargamos el contenido de las URLs
print("\n" + "="*50)
print("FASE 1: CARGA DE SITIOS WEB")
print("="*50)

web_start_time = time.time()
web_docs_count = 0

#filtramos URLs ya procesadas
urls_a_procesar = [u for u in urls if u not in processed_files]

print(f"URLs totales: {len(urls)}")
print(f"Ya procesadas: {len(urls) - len(urls_a_procesar)}")

#cargamos el contenido de las URLs
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

#guardamos los datos de la web en la base de datos
if current_batch:
    print("Guardando datos de la web en la base de datos...")
    vectorstore = process_save_batch(current_batch, vectorstore, total_processed)
    
    #actualizamos historial con las URLs que acabamos de procesar
    save_history(urls_a_procesar)
    current_batch = [] #limpiamos la lista para que no se mezcle con los PDFs
            
#descargamos los PDFs
print("\n" + "="*50)
print("FASE 2: DESCARGA DE PDFs")
print("="*50)
try:
    df_pdfs = pd.read_excel(ARCHIVO_ENLACES_PDF, header=None)
    lista_pdfs = df_pdfs[0].dropna().tolist()
    
    descargar_pdfs(lista_pdfs)
    
except Exception as e:
    print(f"Error al descargar PDFs: {e}")

#procesamos los PDFs locales
print("\n" + "="*50)
print("FASE 3: PROCESAMIENTO DE PDFs")
print("="*50)

carpeta_pdfs = "pdfs_descargados_mini"
if os.path.exists(carpeta_pdfs):
    archivos_pdf = [f for f in os.listdir(carpeta_pdfs) if f.lower().endswith('.pdf')]
    
    #filtramos los que ya estan procesados
    archivos_nuevos = [f for f in archivos_pdf if f not in processed_files]
    
    print(f"Total PDFs encontrados: {len(archivos_pdf)}")
    print(f"Ya procesados: {len(archivos_pdf) - len(archivos_nuevos)}")
    print(f"Nuevos PDFs: {len(archivos_nuevos)}")
   
    #lista para identificar qué archivos están en el lote actual
    current_batch_filenames = []
    
    for i, filename in enumerate(archivos_nuevos):
        filepath = os.path.join(carpeta_pdfs, filename)
        try:
            print(f"[{i+1}/{len(archivos_nuevos)}] Leyendo: {filename} ...", end=" ", flush=True)
            t_load_start = time.time()
            
            #usamos PyPDFLoader de LangChain para cargar los PDFs
            loader_pdf = PyPDFLoader(filepath)
            docs_pdf = loader_pdf.load()
            
            #establecemos el nombre del archivo y añadimos al batch
            for doc in docs_pdf:
                doc.metadata['source'] = filename 
                current_batch.append(doc)
            
            current_batch_filenames.append(filename)
            
            t_load_end = time.time()
            print(f"OK ({round(t_load_end - t_load_start, 2)}s).Buffer: {len(current_batch)} docs")
            
            total_processed += 1
            
            #guardamos si alcanzamos el limite (5 archivos o 200 paginas)
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
#configuracion de retriever
retriever = vectorstore.as_retriever()

#prompt del sistema
system_prompt = (
    "Eres un asistente experto en turismo y medio ambiente de Andalucia. "
    "Responde a las preguntas basandote unicamente en el contexto proporcionado a continuacion. "
    "Si no conoces la respuesta, indicalo claramente. "
    "Responde siempre en español." 
    "\n\n"
    "{context}"
)

#creamos el prompt con el mensaje del sistema y la entrada del usuario
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

#creamos la cadena de documentos que combina los textos
question_answer_chain = create_stuff_documents_chain(llm, prompt)
#creamos la cadena final que une el buscador con el generador de respuestas
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ==========================================
# MODO TEST Y CHAT
# ==========================================
#Creamos el modo test
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
                resp = rag_chain.invoke({"input": p})['answer']
                dt = round(time.time() - t0, 4)
                resultados.append({
                    "Pregunta": p, "Iteración": i, "Tiempo (s)": dt,
                    "Tokens Entrada": int(len(p.split())*1.3), "Tokens Salida": int(len(resp.split())*1.3), "Respuesta": resp
                })
            except Exception as e: print(f"Error {i}: {e}")

    if resultados:
        pd.DataFrame(resultados).to_excel(ARCHIVO_SALIDA, index=False)
        print(f"Resultados guardados en {ARCHIVO_SALIDA}")
except Exception as e: print(f"Error test: {e}")

print("Puede realizar sus consultas sobre Andalucia. (Escriba exit' o 'stop' para salir)\n")

#bucle de chat
while True:
    print("\n" + "="*60)
    query = input("\n INGRESE SU CONSULTA: ")
    #cuando salimos del bucle
    if query.lower() in ["exit", "stop"]:
        break
    
    print("Pensando...")
    try:
        #ejecutamos la cadena RAG con la pregunta del usuario
        response = rag_chain.invoke({"input": query})
        
        print("\n" + "="*60)
        print("RESPUESTA GENERADA:")
        print("="*60)
        print(response['answer'])
        print("\n" + "-"*60)
        
    except Exception as e:
        print(f"[ERROR]: {e}")