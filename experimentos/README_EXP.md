# 2025-tfm-naturcor-anastasia

**Experimento: Chunk Size (basado en v2.2)**

Se realiza una serie de experimentos para medir el efecto del tamaño de chunk en la calidad del modelo. Para aislar la variable chunk_size, se fija el contexto total en 10 000 caracteres ajustando k: 
- CS-500:  chunk_size = 500,   chunk_overlap = 100,  k=20
- CS-1000: chunk_size = 1000,  chunk_overlap = 200,  k=10 - BASELINE
- CS-2000: chunk_size = 2000,  chunk_overlap = 400,  k=5

Cada configuración se lanza sobre 60 preguntas * 10 iteraciones con RAGAS.
La configuración CS-500 obtiene el mejor Faithfulness (0.437) y ContextPrecision (0.929). Las otras dos configuraciones están en ~0.36 y ~0.85 respectivamente. ResponseRelevancy es casi igual en los tres casos (0.61 - 0.62).


**Experimento: Retriver Híbrido (basado en v2.2)**

Se realiza una serie de experimentos para medir el efecto del balance entre la búsqueda semántica (vectorial) y la búsqueda léxica (BM25) en la calidad del modelo. Se combinan ambos retrievers con un ensemble con pesos, manteniendo fijo el contexto total en 10 000 caracteres y la configuración baseline. Para aislar la variable del peso de fusión, solo se varía la combinación de pesos semántico/BM25:

- H-50/50: semantic 0.5 y k=5 / BM25 0.5 y k=5
- H-30/70: semantic 0.3 y k=5 / BM25 0.7 y k=5
- H-70/30: semantic 0.7 y k=5 / BM25 0.3 y k=5

Cada configuración se lanza sobre 60 preguntas * 10 iteraciones con RAGAS. La configuración H-50/50 obtiene el mejor Faithfulness (0.472). Las configuraciones H-30/70 y H-70/30 quedan por debajo (0.456 y 0.435 respectivamente). En ContextPrecision los resultados se invierten: H-70/30 obtiene el mejor valor (0.861), seguida de H-50/50 (0.834) y H-30/70 (0.809). ResponseRelevancy es casi igual en las tres configuraciones (0.60–0.62). En tiempo de RAG por pregunta, las tres configuraciones son rápidas y estables (~6.4-6.6s).

**Experimento: Multi Query (basado en v2.2)**

Se realiza una serie de experimentos para medir el efecto del Multi Query Retriever, que usa el LLM para reformular la pregunta original en varias versiones alternativas, lanzar una búsqueda independiente por cada una y combinar los fragmentos eliminando duplicados. Se mantiene fijo el contexto total en ~10 000 caracteres y la configuración baseline. Para aislar la variable, se varía el número de subqueries generadas y la k de cada búsqueda, manteniendo constante el número de chunks brutos antes de deduplicar (12), suponiendo que la deduplicación reduce este número hasta los 10.

- MQ-B3x4: 3 subqueries / k=4 por subquery
- MQ-B4x3: 4 subqueries / k=3 por subquery

Cada configuración se lanza sobre 60 preguntas * 10 iteraciones con RAGAS. Generar más subqueries con menos profundidad (MQ-B4x3) resulta ligeramente mejor que la configuración opuesta: alcanza 0.389 de Faithfulness y 0.854 de ContextPrecision, frente a 0.360 y 0.817 de MQ-B3x4. ResponseRelevancy y el tiempo de RAG casi no cambian entre las configuraciones (0.61 y ~7.7s). 

**Experimento: Rewrite-Retrieve-Read (basado en v2.2)**

Se realiza una serie de experimentos para medir el efecto de la arquitectura Rewrite-Retrieve-Read en la calidad del modelo. Esta arquitectura híbrida añade un paso previo a la generación: un retrieval inicial con la pregunta original, una reformulación del query por parte del LLM apoyándose en los fragmentos ya recuperados, y un segundo retrieval con la pregunta reformulada (k_inicial=20, k_final=10). Se mantienen fijos el contexto final en 10 000 caracteres y la configuración baseline. Para aislar el efecto, se varían el contexto que recibe el LLM en el paso de reformulación y el prompt usado para reformular:

- RRR-AD-300: primeros 300 caracteres de los top 6 chunks, prompt de adaptación al lenguaje del corpus
- RRR-AD:     top 10 chunks enteros, prompt de adaptación al lenguaje del corpus
- RRR-GE:     top 10 chunks enteros, prompt de concreción geográfica (identifica provincia, parque, sierra y parámetros)

Cada configuración se lanza sobre 60 preguntas * 10 iteraciones con RAGAS. La configuración RRR-GE obtiene el mejor Faithfulness (0.436) y ContextPrecision (0.841) entre las tres variantes RRR, superando a las dos configuraciones con prompt de adaptación (Faithfulness ~0.40, ContextPrecision 0.83-0.87). ResponseRelevancy es similar en las tres configuraciones (0.60-0.62). 

**Experimento: Metadatos (basado en v2.2)**

Se realiza un experimento para medir el efecto de inyectar metadatos estructurados en los chunks antes del embedding. Se extraen 5 campos de la primera página de cada PDF con expresiones regulares (provincia, municipios, dificultad, tipo_trayecto, autorización) y se anteponen como cabecera al inicio de cada chunk con el formato [Provincia: X | Municipios: Y | Dificultad: Z | ...]. Los PDFs con texto extraíble menos de 500 caracteres no se extrayen. Se mantiene la configuración baseline y se crea la base vectorial FAISS con los chunks incluyendo metadatos.

- META-B: cabecera con 5 campos regex añadidos en cada chunk

Cada configuración se lanza sobre 60 preguntas * 10 iteraciones con RAGAS. La configuración META-B obtiene un Faithfulness ligeramente superior al baseline (0.372 frente a ~0.36), pero empeora en ContextPrecision (0.766 frente a ~0.85) y en ResponseRelevancy (0.589 frente a ~0.62). 
