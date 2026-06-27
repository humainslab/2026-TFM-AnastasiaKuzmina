# ==========================================
# IMPORT DE LIBRERÍAS
# ==========================================
import os
import time
import pandas as pd

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
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.output_parsers import StrOutputParser

# formato de datos para evaluar una pregunta-respuesta
from ragas.dataset_schema import SingleTurnSample
# métricas que funcionan sin ground truth
from ragas.metrics import (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference)
# wrappers para utilizar Ollama como LLM-juez
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ==========================================
# CONFIGURACIÓN GENERAL Y SERVIDOR
# ==========================================
# configuracion del test
ARCHIVO_PREGUNTAS = "Escenarios_medio.xlsx"
NUM_REPETICIONES = 10

# nombre de la carpeta donde se va a guardar la base de datos
DB_NAME = "andalucia_db"

#TOKEN
TOKEN = ''
ollama_server = os.environ.get("OLLAMA_SERVER", "")
headers = {"authorization": f"Bearer {TOKEN}"}

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

# ==========================================
# CONFIGURACIÓN DEL LLM-JUEZ PARA RAGAS
# ==========================================

evaluator_llm = LangchainLLMWrapper(llm)

evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)


# numero de afirmaciones verificadas por el contexto / numero total de afirmaciones
metric_faithfulness = Faithfulness(llm=evaluator_llm)

# promedio de la similitud entre la pregunta original y las preguntas generadas a partir de la respuesta
metric_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

# media ponderada de la relevancia de los chunks según su posición
metric_context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

# agrupamos todas las métricas en un diccionario
ragas_metrics = {
    "Faithfulness": metric_faithfulness,
    "ResponseRelevancy": metric_relevancy,
    "ContextPrecision": metric_context_precision,
}

# ==========================================
# FUNCIÓN DE EVALUACIÓN DE UNA RESPUESTA
# ==========================================

# creamos una función para evaluar una respuesta con todas las métricas RAGAS
def evaluate_single(question, answer, contexts):
    # formato de datos de RAGAS
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    
    scores = {}
    # recorremos cada métrica y calculamos la puntuación
    for name, metric in ragas_metrics.items():
        try:
            score = metric.single_turn_score(sample)
            scores[name] = round(float(score), 4)
        except Exception as e:
            print(f"Error al calcular {name}: {e}")
            scores[name] = None
    
    return scores


# ==========================================
# ESTADÍSTICAS RESUMIDAS
# ==========================================
def print_statistics(df_results):
    print("\n")
    print("ESTADÍSTICAS RESUMIDAS")
    print("-" * 60)

    for metric_name in ["RAGAS Faithfulness", "RAGAS ResponseRelevancy", "RAGAS ContextPrecision"]:
        # tomamos solo los valores no vacíos
        values = df_results[metric_name].dropna()
        if len(values) > 0:
            print(f"  {metric_name}:")
            print(f"    Media:    {values.mean():.4f}")
            print(f"    Mediana:  {values.median():.4f}")
            print(f"    Mín:      {values.min():.4f}")
            print(f"    Máx:      {values.max():.4f}")
            print(f"    Desv.est: {values.std():.4f}")
        else:
            print(f"  {metric_name}: no hay datos")

    # tiempo de evaluación RAGAS
    if "Tiempo RAGAS (s)" in df_results.columns:
        ragas_times = df_results["Tiempo RAGAS (s)"].dropna()
        if len(ragas_times) > 0:
            print(f"\n  Tiempo medio RAGAS: {ragas_times.mean():.4f}s")

    # contar evaluaciones exitosas
    ok_count = df_results["RAGAS Faithfulness"].notna().sum()
    print(f"  Evaluaciones OK:    {ok_count}/{len(df_results)}")


# ==========================================
# MODO 1: EVALUACIÓN DE RESPUESTAS (desde excel)
# ==========================================
def run_mode_1():
    # pedimos el nombre del archivo de entrada
    print("\nIntroduzca el nombre del archivo Excel con las respuestas ya generadas")
    print("(debe estar en la misma carpeta que este script)")
    archivo_entrada = input("Nombre del archivo: ").strip()
    
    # comprobamos que el archivo existe
    if not os.path.exists(archivo_entrada):
        print(f"ERROR: No se encontró el archivo '{archivo_entrada}'")
        return
    
    # nombre del archivo de salida
    nombre_base = os.path.splitext(archivo_entrada)[0]
    archivo_salida = f"{nombre_base}_ragas.xlsx"
    
    # cargamos el archivo
    print(f"\nCargando '{archivo_entrada}'...", end=" ", flush=True)
    try:
        df = pd.read_excel(archivo_entrada, engine='openpyxl')
        print(f"OK ({len(df)} filas)")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # comprobamos que tiene las columnas necesarias
    columnas_necesarias = ['Pregunta', 'Respuesta', 'Contexto usado']
    for col in columnas_necesarias:
        if col not in df.columns:
            print(f"ERROR: Falta la columna '{col}' en el archivo")
            print(f"Columnas encontradas: {list(df.columns)}")
            return
    
    # bucle de evaluación
    print(f"\n{'='*60}")
    print("INICIO DE LA EVALUACIÓN (MODO 1 - respuestas de excel)")
    print(f"Filas a evaluar: {len(df)}")
    print(f"Métricas RAGAS: {', '.join(ragas_metrics.keys())}")
    print(f"{'='*60}")
    
    # creamos las columnas nuevas para las puntuaciones
    df["RAGAS Faithfulness"] = None
    df["RAGAS ResponseRelevancy"] = None
    df["RAGAS ContextPrecision"] = None
    df["Tiempo RAGAS (s)"] = None
    
    for idx, row in df.iterrows():
        pregunta = str(row['Pregunta'])
        respuesta = str(row['Respuesta'])
        contexto = str(row['Contexto usado'])
        
        # mostramos el progreso
        iteracion = row.get('Iteración', '')
        iter_str = f" (iteración {int(iteracion)})" if pd.notna(iteracion) else ""
        print(f"\n[{idx+1}/{len(df)}] {pregunta[:50]}...{iter_str}")
        
        # pasamos el contexto como lista de un elemento
        context_list = [contexto]
        
        # evaluamos con RAGAS
        print(f"Calculando métricas RAGAS...")
        t1 = time.time()
        ragas_scores = evaluate_single(pregunta, respuesta, context_list)
        dt_ragas = round(time.time() - t1, 4)
        
        print(f"RAGAS: {dt_ragas}s | Puntuaciones: {ragas_scores}")
        
        # guardamos las puntuaciones en el DataFrame
        df.at[idx, "RAGAS Faithfulness"] = ragas_scores.get("Faithfulness")
        df.at[idx, "RAGAS ResponseRelevancy"] = ragas_scores.get("ResponseRelevancy")
        df.at[idx, "RAGAS ContextPrecision"] = ragas_scores.get("ContextPrecision")
        df.at[idx, "Tiempo RAGAS (s)"] = dt_ragas
        
        # guardamos el excel después de cada fila
        df.to_excel(archivo_salida, index=False)
    
    print(f"\n{'='*60}")
    print("RESULTADOS FINALES")
    print(f"{'='*60}")
    
    df.to_excel(archivo_salida, index=False)
    print(f"Resultados guardados en: {archivo_salida}")
    
    print_statistics(df)

# ==========================================
# MODO 2: GENERACIÓN + EVALUACIÓN
# ==========================================
def run_mode_2():
    archivo_salida = "resultados_ragas_RRR-AD.xlsx"
    
    print(f"\n{'='*60}")
    print("CARGA DE LA BASE DE DATOS")
    print(f"{'='*60}")
    
    if not os.path.exists(DB_NAME):
        print(f"ERROR: La base de datos '{DB_NAME}' no encontrada")
        print("Ejecute el bot principal para crear la base de datos.")
        return
    
    print(f"Cargando base de datos '{DB_NAME}'...", end=" ", flush=True)
    t_start = time.time()
    vectorstore = FAISS.load_local(DB_NAME, embeddings, allow_dangerous_deserialization=True)
    print(f"OK ({round(time.time() - t_start, 2)}s)")
    
    # prompt para reformular la pregunta según el lenguaje del corpus
    rewrite_prompt_rrr = ChatPromptTemplate.from_messages([
        ("system",
         "Eres un asistente que adapta preguntas al lenguaje técnico de documentos de senderismo andaluz. "
         "Dado el contexto recuperado y la pregunta original, reformula la pregunta usando "
         "los términos y estilo del contexto para mejorar la búsqueda. "
         "Devuelve ÚNICAMENTE la pregunta reformulada, sin explicaciones ni introducción."),
        ("human", "Pregunta original: {question}\n\nContexto recuperado:\n{context}\n\nPregunta reformulada:"),
    ])

    class RRRRetriever(BaseRetriever):
        """Rewrite-Retrieve-Read."""
        vectorstore: object
        llm: object
        k_inicial: int = 20
        k_final: int = 10

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query, *, run_manager: CallbackManagerForRetrieverRun):
            # 1 - recuperación inicial con k alto
            rough_docs = self.vectorstore.as_retriever(search_kwargs={"k": self.k_inicial}).invoke(query)
            # 2 - reformulación de la pregunta según el contexto recuperado
            contexto_rough = "\n\n".join(doc.page_content for doc in rough_docs[:10])
            rewrite_chain = rewrite_prompt_rrr | self.llm | StrOutputParser()
            pregunta_reformulada = rewrite_chain.invoke({"question": query, "context": contexto_rough})
            # 3 - recuperación final con la pregunta reformulada
            return self.vectorstore.as_retriever(search_kwargs={"k": self.k_final}).invoke(pregunta_reformulada)

    # configuracion de retriever RRR
    retriever = RRRRetriever(vectorstore=vectorstore, llm=llm)

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
    "{context}")
    
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
    

    print(f"\n{'='*60}")
    print("CARGA DE PREGUNTAS")
    print(f"{'='*60}")
    
    try:
        df_preg = pd.read_excel(ARCHIVO_PREGUNTAS, header=None, engine='openpyxl')
        preguntas = [str(p).strip().replace('"', '') for p in df_preg[0].dropna() if len(str(p)) > 3]
        print(f"Preguntas cargadas: {len(preguntas)}")
    except Exception as e:
        print(f"ERROR al cargar las preguntas: {e}")
        return
    
    # bucle principal
    print(f"\n{'='*60}")
    print("INICIO DE LA EVALUACIÓN (MODO 2 - generación + evaluación)")
    print(f"Preguntas: {len(preguntas)}, repeticiones: {NUM_REPETICIONES}")
    print(f"Total de iteraciones: {len(preguntas) * NUM_REPETICIONES}")
    print(f"Métricas RAGAS: {', '.join(ragas_metrics.keys())}")
    print(f"{'='*60}")
    
    resultados = []
    total_iterations = len(preguntas) * NUM_REPETICIONES
    current_iteration = 0
    
    for p in preguntas:
        for i in range(1, NUM_REPETICIONES + 1):
            current_iteration += 1
            print(f"\n[{current_iteration}/{total_iterations}] {p[:50]}... (iteración {i})")
            
            t0 = time.time()
            try:
                # obtenemos la respuesta del RAG
                # en modo TEST enviamos historial vacío porque cada pregunta es independiente
                result = rag_chain.invoke({"input": p, "chat_history": []})
                resp = result['answer']
                context_docs = result.get('context', [])

                # extraemos el texto del contexto y las fuentes
                context_texts = [doc.page_content for doc in context_docs]
                context_text_combined = " ".join(context_texts)
                fuentes = set([doc.metadata.get('source', 'Desconocido') for doc in context_docs])
                fuentes_str = ", ".join(fuentes)

                dt_rag = round(time.time() - t0, 4)
                print(f"  RAG: {dt_rag}s | Respuesta: {len(resp)} caracteres | Chunks: {len(context_texts)}")

                # evaluamos con RAGAS
                print(f"Calculando métricas RAGAS...")
                t1 = time.time()
                ragas_scores = evaluate_single(p, resp, context_texts)
                dt_ragas = round(time.time() - t1, 4)
                print(f"RAGAS: {dt_ragas}s | Puntuaciones: {ragas_scores}")

                # recopilamos los datos
                input_text = p + " " + context_text_combined
                resultados.append({
                    "Pregunta": p,
                    "Iteración": i,
                    "Tiempo RAG (s)": dt_rag,
                    "Tokens Entrada": int(len(input_text.split()) * 1.3),
                    "Tokens Salida": int(len(resp.split()) * 1.3),
                    "Respuesta": resp,
                    "Fuentes": fuentes_str,
                    "Contexto usado": context_text_combined,
                    "RAGAS Faithfulness": ragas_scores.get("Faithfulness"),
                    "RAGAS ResponseRelevancy": ragas_scores.get("ResponseRelevancy"),
                    "RAGAS ContextPrecision": ragas_scores.get("ContextPrecision"),
                    "Tiempo RAGAS (s)": dt_ragas,
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                resultados.append({
                    "Pregunta": p,
                    "Iteración": i,
                    "Tiempo RAG (s)": round(time.time() - t0, 4),
                    "Tokens Entrada": None,
                    "Tokens Salida": None,
                    "Respuesta": f"ERROR: {e}",
                    "Fuentes": None,
                    "Contexto usado": None,
                    "RAGAS Faithfulness": None,
                    "RAGAS ResponseRelevancy": None,
                    "RAGAS ContextPrecision": None,
                    "Tiempo RAGAS (s)": None,
                })
            
            # guardamos el excel después de cada fila
            pd.DataFrame(resultados).to_excel(archivo_salida, index=False)
    

    print(f"\n{'='*60}")
    print("RESULTADOS FINALES")
    print(f"{'='*60}")
    
    df_results = pd.DataFrame(resultados)
    df_results.to_excel(archivo_salida, index=False)
    print(f"Resultados guardados en: {archivo_salida}")
    
    print_statistics(df_results)
    
# ==========================================
# SELECCIÓN DE MODO
# ==========================================
print("\n" + "=" * 60)
print("EVALUACIÓN CON RAGAS")
print("=" * 60)
print("1 - Evaluar respuestas (desde archivo Excel)")
print("El archivo debe tener: Pregunta, Respuesta, Contexto usado")
print()
print("2 - Generar respuestas con RAG y evaluarlas")
print("=" * 60)

while True:
    modo = input("\nSeleccione modo (1 o 2): ").strip()
    if modo in ["1", "2"]:
        break
    print("Por favor, introduzca 1 o 2")

if modo == "1":
    run_mode_1()
elif modo == "2":
    run_mode_2()
