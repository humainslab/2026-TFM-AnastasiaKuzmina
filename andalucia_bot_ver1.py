#importamos las librerías necesarias
import os
import sys
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

#nombre de la carpeta donde se va a guardar la base de datos
DB_NAME = "andalucia_db"

#lista de URLs
urls = [
    "https://www.juntadeandalucia.es/medioambiente/portal/landing-page-%C3%ADndice/-/asset_publisher/zX2ouZa4r1Rf/content/oficinas-y-centros-de-visitantes-de-espacios-naturales-protegidos-de-andaluc-c3-ada/20151",
    "https://www.juntadeandalucia.es/medioambiente/portal/web/ventanadelvisitante/inicio"
]




#convierte el texto en listas de números para que el ordenador pueda buscar información
embeddings = OllamaEmbeddings(model="llama3:8b")
#el modelo que lee la información y responde
llm = ChatOllama(model="llama3:8b")

#comprobamos si la base de datos ya existe
if os.path.exists(DB_NAME):
    print(f"\nBase de datos '{DB_NAME}' ya existe")
#si existe, cargamos desde el disco
    vectorstore = FAISS.load_local(DB_NAME, embeddings, allow_dangerous_deserialization=True)
else:
    print("Base de datos no encontrada. Cargamos los datos...")
    
#cargamos el contenido de las URLs
    loader = WebBaseLoader(
        urls,
        header_template={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    )
    docs = loader.load()

#dividimos el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

#convertimos el texto en vectores para que el modelo pueda buscar por significado
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
#guardamos la base de datos
    vectorstore.save_local(DB_NAME)
    print(f"Base de datos guardada en la carpeta '{DB_NAME}'")

#convertimos la base de datos en un retriever
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

print("Puede realizar sus consultas sobre Andalucia. (Escriba 'exit' o 'stop' para salir)\n")

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
        print(response['answer'])
        
    except Exception as e:
        print(f"[ERROR]: {e}")