# 2026-TFM-AnastasiaKuzmina
Repositorio para TFM

v1
se implementa un bot RAG básico sobre las rutas ecológicas en Andalucia. Se basa en contenido desde un conjunto fijo de URLs, lo fragmenta, genera embeddings con Ollama (llama3:8b) y crea/carga una base vectorial FAISS en disco. La busqueda se hace con un retriever + prompt restringido al contexto y un bucle interactivo por consola. 


v2
Se añade actualización de la FAISS (carga + add_documents + guardado por lotes), control de elementos ya procesados mediante procesados.txt, y nuevas fuentes: descarga de PDFs desde enlaces en Excel y lectura con PyPDFLoader. Además se incorpora un modo de test automático con preguntas desde Excel, repeticiones, medición de tiempos y exportación de resultados a un Excel, manteniendo luego el modo chat.


v2.1
Se añade un token de autenticación para la conexión con el servidor Ollama. Se modifica el parámetro de búsqueda del retriever a k=10. Se actualiza el system prompt para permitir el uso de conocimiento general en preguntas de ubicación geográfica, limitando el resto de consultas estrictamente al contexto. Se corrige la fórmula de cálculo de tokens de entrada en el modo test para incluir el contexto inyectado por FAISS. Se añaden las columnas con la fuente y el contexto utilizado en la exportación de resultados a Excel.


v2.2 Se implementa memoria conversacional (limitada a 10 mensajes). Además, se actualiza el system prompt para que el chatbot sea más preciso (fuentes de información, conocimiento general permitido, filtro geográfico, formato de respuesta) y se añade un menú de selección de modo (test y chat).

Se implementa un evaluador basado en RAGAS que calcula tres métricas, usando Ollama como LLM juez, sin necesidad de ground truth: 
- Faithfulness (si la respuesta se basa en el contexto recuperado)
- Response Relevancy (si la respuesta es relevante a la pregunta)
- Context Precision (si los fragmentos relevantes están en posiciones superiores en el ranking de recuperación)

