# Plan sustentado para la primera etapa de modelado

Proyecto 1 - IEEE-CIS Fraud Detection

Este documento resume el plan metodologico para iniciar la parte de creacion del modelo del sistema adaptativo de deteccion de fraude. El plan se basa en cuatro fuentes de evidencia:

- El planteo del proyecto en `p1/00_brief/Project 1.pdf`.
- El EDA del companero: `fabryzziomezatorres/ieee-fraud-eda-riesgos`.
- El EDA propio ejecutado en Kaggle: `biancaaguinaga/p1-ieee-fraud-eda`.
- Literatura sobre fraude, datos desbalanceados, validacion temporal y concept drift.

El objetivo de esta primera etapa no es maximizar inmediatamente el leaderboard, sino construir una base experimental defendible, alineada con el proyecto: deteccion de fraude, toma de decision basada en riesgo, monitoreo temporal y adaptacion ante cambios.

## 1. Contexto del problema

El dataset IEEE-CIS Fraud Detection contiene transacciones electronicas con variables de transaccion, tarjeta, identidad, dispositivo y variables anonimizadas generadas por Vesta. La variable objetivo es `isFraud`, con valor 1 para fraude y 0 para transaccion legitima.

El planteo del proyecto define cuatro objetivos:

1. Detectar transacciones fraudulentas.
2. Transformar el riesgo estimado en acciones de aprobacion, revision o escalamiento.
3. Detectar degradacion temporal del desempeno.
4. Evaluar una estrategia adaptativa de reentrenamiento.

Por lo tanto, el modelo no debe evaluarse solo como clasificador estatico. Debe ser tratado como un componente dentro de un sistema de decision que opera en el tiempo.

## 2. Evidencia del EDA

El EDA propio se ejecuto en Kaggle como kernel publico:

`biancaaguinaga/p1-ieee-fraud-eda`

Los outputs descargados estan en:

`p1/03_outputs/eda/`

Los archivos principales son:

- `eda_report.md`
- `eda_summary.json`
- `decision_log.csv`
- `missing_summary.csv`
- `numeric_shift_ks.csv`
- `uid_overlap_analysis.csv`
- `target_correlations.csv`
- `temporal_summary.csv`
- `category_risk.csv`

### 2.1 Tamano y distribucion de clases

El conjunto de entrenamiento tiene 590,540 filas y el conjunto de prueba tiene 506,691 filas. La tasa de fraude en train es 3.50%.

Esto confirma un problema de clasificacion binaria fuertemente desbalanceado. En este contexto, una metrica como accuracy puede ser enganosa: un modelo que prediga siempre "no fraude" tendria una accuracy cercana a 96.5%, pero no detectaria ningun fraude.

Segun Krawczyk (2016), los problemas desbalanceados suelen sesgar los modelos hacia la clase mayoritaria, aunque la clase minoritaria sea la mas importante desde el punto de vista operativo. En fraude, esa clase minoritaria representa justamente el evento de mayor interes. Por eso, la evaluacion debe priorizar metricas sensibles a la clase positiva.

Decision:

- Usar PR-AUC como metrica principal.
- Reportar tambien Recall, Precision, F1, ROC-AUC y FPR.
- No usar accuracy como metrica de seleccion.

Justificacion:

Saito y Rehmsmeier (2015) muestran que las curvas Precision-Recall son mas informativas que ROC cuando se evaluan clasificadores binarios en datasets desbalanceados. Esto aplica directamente al caso IEEE-CIS, porque el fraude representa solo 3.50% del train.

### 2.2 Variacion temporal y no estacionariedad

El EDA propio encontro que la tasa semanal de fraude varia entre 1.85% y 5.06%. Es decir, la frecuencia de fraude no es estable a lo largo del tiempo.

Esto tiene dos implicancias:

- El desempeno promedio global puede ocultar periodos de degradacion.
- Un split aleatorio mezclaria pasado y futuro, produciendo una estimacion demasiado optimista.

La literatura sobre concept drift define el problema como un cambio en la relacion entre las variables de entrada y la variable objetivo a lo largo del tiempo. Gama et al. (2014) explican que los sistemas adaptativos deben considerar cambios temporales y metodologias de evaluacion especificas para escenarios no estacionarios.

Decision:

- Ordenar siempre por `TransactionDT`.
- Usar validacion temporal, no split aleatorio.
- Medir metricas por ventana temporal.
- Evaluar modelo estatico vs modelo adaptativo en etapas posteriores.

Justificacion:

La documentacion de `TimeSeriesSplit` de scikit-learn indica que, en datos ordenados temporalmente, tecnicas como `KFold` o `ShuffleSplit` pueden producir estimaciones poco realistas porque mezclan observaciones correlacionadas cercanas en el tiempo y permiten evaluar sobre observaciones que no representan verdaderamente el futuro.

### 2.3 Separacion train/test y riesgo de usar tiempo crudo

El adversarial validation train/test del EDA propio obtuvo ROC-AUC = 1.000. La importancia de permutacion mostro que la variable dominante para separar train de test fue `DT_day`.

Interpretacion:

- Train y test pertenecen a periodos temporales consecutivos.
- El modelo puede distinguir train de test casi perfectamente usando posicion temporal.
- Esto no es necesariamente leakage del target, sino una caracteristica del diseno de la competencia.
- Pero usar `TransactionDT` crudo como feature predictiva haria que el modelo aprenda "posicion en el calendario" mas que patrones robustos de fraude.

Decision:

- No usar `TransactionDT` crudo como predictor.
- Usarlo para ordenar, particionar y monitorear.
- Derivar variables temporales controladas como hora o semana solo si aportan interpretabilidad operacional y se validan temporalmente.

Justificacion:

El proyecto exige simular evaluacion sobre datos futuros. Si una variable codifica directamente el avance temporal entre train y test, puede mejorar metricas offline sin representar una regla generalizable de fraude.

### 2.4 UIDs y diferencia entre clientes conocidos y desconocidos

El EDA del companero propuso un cliente candidato basado en:

`card1 + card2 + card3 + card5 + addr1 + addr2 + D1 ajustado por dia`

El EDA propio confirmo que, usando un corte temporal en el percentil 80 de `TransactionDT`, 42.76% de las filas tardias comparten candidate UID con el entrenamiento temprano.

Tambien se observo que la tasa de fraude difiere entre segmentos:

- UID conocido tardio: 2.01%.
- UID desconocido tardio: 4.51%.

Interpretacion:

El problema no es homogeneo. Las transacciones de clientes ya vistos y clientes nuevos tienen comportamientos de riesgo distintos. Un modelo puede parecer fuerte si aprende historiales de UIDs conocidos, pero fallar en UIDs nuevos, que son mas dificiles y tienen mayor tasa de fraude.

Decision:

- Crear features basadas en UID, pero calcularlas solo con informacion pasada.
- Reportar desempeno separado para:
  - validacion global;
  - UID conocido;
  - UID desconocido.
- Evitar target encoding o agregaciones que usen informacion del futuro.

Justificacion:

La decision se sostiene en la estructura observada del dataset, no en la imitacion de una solucion externa. El EDA muestra que una gran proporcion de transacciones comparte un mismo cliente candidato y que el riesgo cambia entre UIDs conocidos y desconocidos. En un sistema real, una transaccion no se evalua en aislamiento: si una tarjeta, direccion, dominio o combinacion de atributos ya aparecio antes, su historial operacional es informacion disponible al momento de decidir. Por eso, las features por UID son validas siempre que se calculen de forma historica, usando solo informacion anterior a la transaccion evaluada.

El criterio es operacional: usar memoria historica permitida sin mirar el futuro. La restriccion metodologica es que ninguna frecuencia, promedio, conteo o agregacion puede usar filas de validacion o de una ventana posterior. Asi se evita que el modelo aprenda informacion que no estaria disponible en produccion.

### 2.5 Valores faltantes

El EDA del companero encontro que 75.6% de las filas no tienen fila asociada en `train_identity`. El EDA propio encontro 74 features con mas de 80% de faltantes en train.

Ademas, varias columnas `V` faltan en bloques. Esto sugiere que los faltantes no son completamente aleatorios, sino estructurales o condicionados por variables observables como `ProductCD`.

Interpretacion:

La ausencia de datos puede contener senal. Por ejemplo, no tener informacion de identidad puede corresponder a un canal, producto o flujo de autenticacion distinto.

Decision:

- Crear `has_identity`.
- Crear indicadores de faltante por columna o por bloque.
- No imputar todo mecanicamente con media.
- Para modelos de arboles, permitir que el modelo maneje faltantes cuando sea posible.
- Para modelos que requieren imputacion, usar estrategias ajustadas solo en train.

Justificacion:

En datos transaccionales reales, la ausencia de informacion muchas veces describe el proceso que genero la transaccion. Tratarla como error aleatorio puede destruir senal util.

### 2.6 TransactionAmt y outliers

El EDA propio encontro:

- `TransactionAmt` tiene skew = 14.37.
- El percentil 50 es 68.77.
- El percentil 95 es 445.00.
- El percentil 99 es 1104.00.
- Los outliers por IQR representan 11.26% de las filas.
- La tasa de fraude en outliers IQR es 5.07%, mayor que la tasa global.

Interpretacion:

La distribucion de montos es fuertemente asimetrica, lo cual es esperable en pagos digitales: muchas compras pequenas y pocas compras grandes. Pero los montos extremos no parecen ser errores de captura; tienen mayor tasa de fraude.

Decision:

- No eliminar outliers de `TransactionAmt`.
- Crear `log1p(TransactionAmt)`.
- Crear flags de monto extremo.
- Evaluar diferencias entre monto individual y estadisticas historicas del UID.

Justificacion:

Krawczyk (2016) advierte que en problemas desbalanceados los ejemplos raros o extremos de la clase minoritaria pueden ser importantes; eliminarlos puede empeorar la capacidad de detectar casos dificiles. En fraude, una transaccion atipica puede ser justamente la senal relevante.

### 2.7 Variables categoricas y segmentos de riesgo

El EDA propio encontro diferencias de tasa de fraude por categorias:

- `ProductCD = C` tiene lift alto frente a la tasa global.
- `DeviceType = mobile` tiene tasa de fraude superior al promedio.
- Algunos dominios de email como `mail.com`, `outlook.com`, `icloud.com` y `gmail.com` aparecen con lifts altos en ciertos roles (`P_emaildomain` o `R_emaildomain`).
- `card6 = credit` tiene mayor tasa de fraude que el promedio.

Interpretacion:

Estas variables no deben ser interpretadas causalmente porque varias estan anonimizadas o dependen del contexto de Vesta. Sin embargo, estadisticamente ayudan a segmentar riesgo.

Decision:

- Mantener categoricas relevantes.
- Usar encoding adecuado al modelo:
  - CatBoost puede manejar categoricas nativamente.
  - LightGBM/XGBoost requeriran label/frequency encoding.
- Evitar one-hot masivo en columnas de alta cardinalidad.

Justificacion:

El problema es tabular con variables categoricas y numericas anonimizadas. Modelos de boosting son adecuados para este tipo de estructura porque capturan interacciones no lineales y toleran faltantes mejor que modelos lineales simples.

### 2.8 Redundancia en columnas C y V

El EDA del companero y el EDA propio encontraron redundancia fuerte en grupos de columnas, especialmente:

- `C1`, `C2`, `C4`, `C6`, `C7`, `C8`, `C10`, `C11`, `C12`, `C14`.
- Subgrupos de columnas `V` con correlaciones altas.

Interpretacion:

No toda redundancia es necesariamente mala para modelos de arboles, pero puede aumentar costo computacional, dificultar interpretacion y hacer mas inestable la importancia de variables.

Decision:

- En el baseline, conservar la mayoria de columnas para no perder senal prematuramente.
- En la auditoria de features, identificar grupos redundantes.
- Probar una version reducida con representantes por grupo o agregados.

Justificacion:

Para la primera etapa conviene evitar eliminaciones agresivas. La seleccion de features debe basarse en evidencia: importancia, estabilidad temporal, shift train/test y efecto sobre PR-AUC.

## 3. Decisiones para la primera etapa

Las decisiones de esta etapa siguen tres criterios. Primero, evidencia empirica del EDA: distribucion de clases, variacion temporal, faltantes, UIDs, outliers y shift train/test. Segundo, validez operacional: solo se usan variables y transformaciones que podrian estar disponibles al momento de evaluar una transaccion real. Tercero, respaldo metodologico: literatura sobre desbalance, validacion temporal, fraude y concept drift. Las soluciones publicas de Kaggle se consideran antecedentes tecnicos, pero no son la razon principal para adoptar una tecnica.

### Como se aprueba una tecnica

Ninguna tecnica queda aprobada solo porque sea comun en machine learning. Para esta etapa se usara una regla de decision:

1. Debe resolver un problema observado en el EDA.
2. Debe ser compatible con un escenario real de prediccion, es decir, no puede usar informacion futura.
3. Debe tener una forma de verificacion empirica en validacion temporal.
4. Si no mejora PR-AUC, estabilidad temporal, recall util o costo esperado, se descarta o queda solo como analisis exploratorio.

Esto es importante porque varias tecnicas pueden sonar razonables, pero tambien pueden empeorar el modelo. Por ejemplo, aplicar `log1p(TransactionAmt)` puede ayudar a estabilizar una variable muy asimetrica, pero no se debe reemplazar automaticamente el monto original si el modelo de arboles ya lo maneja bien. Por eso se conservara el monto original y se agregara la version transformada como candidata; luego la validacion temporal decidira si aporta.

### Auditoria de decisiones tecnicas

| Decision tecnica | Problema que resuelve | Evidencia del EDA | Criterio de uso | Riesgo | Como se verifica |
| --- | --- | --- | --- | --- | --- |
| Usar PR-AUC como metrica principal | La clase fraude es minoritaria y accuracy puede ser alta aunque el modelo no detecte fraudes | Fraude = 3.50% en train | Priorizar ranking y precision/recall de la clase positiva | Optimizar solo PR-AUC podria ignorar costo operativo | Reportar tambien recall, precision, F1, FPR y costo esperado |
| Split temporal | Evita entrenar con informacion que en produccion perteneceria al futuro | La tasa semanal de fraude cambia entre 1.85% y 5.06% | Entrenar en pasado y validar en futuro | Una sola ventana puede depender de un periodo especifico | Evaluar por varias ventanas temporales |
| No usar `TransactionDT` crudo | Evita que el modelo aprenda posicion temporal en lugar de patrones de fraude | Adversarial train/test ROC-AUC = 1.000 explicado por `DT_day` | Usar tiempo para ordenar y monitorear, no como identificador de periodo | Perder patrones horarios reales | Permitir derivados como hora/dia y medir si generalizan |
| Derivar hora o dia | Captura patrones operativos plausibles, como horarios de mayor riesgo | Existe variacion temporal y `TransactionDT` define secuencia | Usar derivados de baja granularidad, no el timestamp absoluto | Puede sobreajustar a calendario especifico | Comparar modelo con y sin derivados temporales |
| `log1p(TransactionAmt)` | Reduce asimetria y comprime montos extremos sin eliminarlos | Skew de `TransactionAmt` = 14.37; p50 = 68.77, p99 = 1104.00 | Agregar como feature candidata junto al monto original | Puede no aportar en arboles o perder interpretacion directa | Ablation: raw amount vs raw + log amount |
| Flag de outlier de monto | Permite que el modelo identifique montos extremos como condicion especial | Outliers IQR = 11.26%; fraude en outliers = 5.07% | Usarlo como senal binaria, no para eliminar filas | IQR puede marcar compras legitimas caras | Comparar recall/precision en outliers y no-outliers |
| `has_identity` | Representa ausencia estructural del bloque identity | 75.6% sin identity en EDA del companero; missing estructural | Crear indicador y dejar faltantes o imputar segun modelo | El modelo puede aprender un sesgo de canal | Medir metricas por `has_identity` y estabilidad temporal |
| Missing indicators por bloque | Captura patrones de ausencia informativa | 74 features con mas de 80% faltante; columnas V faltan en bloques | Crear flags por grupos de columnas con mismo patron de faltante | Aumenta dimensionalidad | Ablation con y sin flags de missing |
| Frequency encoding | Representa rareza o frecuencia de categorias/UIDs sin usar target | `card1` y variables UID tienen alta cardinalidad | Calcular frecuencias solo en train o ventana pasada | Frecuencias pueden cambiar con drift | Medir estabilidad temporal y PR-AUC por ventana |
| Evitar one-hot masivo en alta cardinalidad | Evita matrices enormes y categorias raras casi unicas | `card1` tiene 13,553 valores unicos | Usar frequency/label encoding o CatBoost | Encoding ordinal puede inducir orden artificial | Comparar con modelo que maneje categorias nativamente |
| UID features historicas | Usa memoria de entidad disponible antes de decidir | 42.76% de filas tardias tienen UID conocido; fraude conocido 2.01%, desconocido 4.51% | Conteos/agregaciones solo con informacion pasada | Leakage si se usan filas futuras | Validacion separada UID conocido/desconocido |
| Agregaciones por UID de monto | Detecta desviaciones frente al comportamiento historico de la entidad | Monto tiene cola pesada y outliers con mayor fraude | Usar media/mediana/desvio historico del UID | Pocos datos por UID pueden dar agregados ruidosos | Suavizar por conteo y medir efecto por UID conocido |
| Reducir redundancia C/V | Baja costo y puede estabilizar importancias | Pares C con correlacion > 0.99; grupos V redundantes | No eliminar en baseline; auditar despues | Eliminar columnas puede perder interacciones utiles | Comparar modelo completo vs reducido |
| Adversarial validation | Detecta si train/test o ventanas son faciles de separar | AUC train/test = 1.000 por tiempo | Usarlo como auditoria de shift, no como metrica final | Puede confundir drift esperado con problema grave | Interpretar importancias: tiempo vs features operativas |
| KS/PSI de variables | Cuantifica cambios de distribucion entre ventanas | Variables V y D15 tienen shift no temporal visible | Monitorear variables clave por ventana | Un shift no siempre degrada performance | Cruzar shift con PR-AUC/Recall por ventana |
| LightGBM baseline | Modelo tabular rapido para muchas columnas, faltantes e interacciones | Dataset grande: 590,540 filas y 439 columnas tras merge | Usarlo como baseline fuerte, no como verdad final | Puede sobreajustar si se permite demasiada complejidad | Early stopping, validacion temporal y comparacion con modelos |

### Glosario operativo de tecnicas

Esta seccion explica las tecnicas en terminos practicos, porque el objetivo no es aplicar nombres de metodos sino entender que problema resuelve cada uno.

**PR-AUC.** Mide que tan bien el modelo ordena los fraudes arriba manteniendo precision. En fraude, no alcanza con decir "detecte muchos fraudes"; tambien importa cuantos falsos positivos se generan para encontrarlos. PR-AUC resume esa tension entre detectar fraude y no saturar al equipo de revision.

**ROC-AUC.** Mide capacidad general de separar positivos y negativos. Se reporta porque es estandar y permite comparar, pero no sera la metrica principal porque con clases muy desbalanceadas puede verse alta aunque la precision sobre fraude sea baja.

**Recall.** De todos los fraudes reales, que porcentaje detectamos. Es importante porque cada fraude no detectado tiene costo.

**Precision.** De todas las transacciones marcadas como sospechosas, cuantas eran fraude. Es importante porque baja precision implica muchas revisiones o bloqueos incorrectos.

**FPR.** De todas las transacciones legitimas, que porcentaje marcamos incorrectamente como fraude. En operacion, esto representa friccion para clientes reales.

**`log1p(TransactionAmt)`.** Es el logaritmo de `1 + monto`. Se usa cuando una variable tiene cola larga: muchos valores pequenos y pocos valores enormes. La transformacion no elimina montos altos; solo reduce la distancia numerica extrema entre ellos. Esto puede hacer que ciertos patrones sean mas faciles de aprender y comparar. Pero como no queremos imponer una transformacion innecesaria, el monto original se conserva.

**Outlier flag.** No significa "borrar outliers". Significa crear una columna binaria que diga si el monto cae en una zona extrema. Esto deja que el modelo aprenda si esa condicion aumenta riesgo. En este dataset, los outliers de monto tienen mas fraude que el promedio, por lo tanto eliminarlos seria perder informacion.

**Frequency encoding.** Reemplaza una categoria por su frecuencia. Por ejemplo, si un dominio de email aparece muchas veces, recibe un valor alto; si aparece pocas veces, recibe un valor bajo. No usa el target, por lo que es menos riesgoso que target encoding. Sirve cuando hay muchas categorias y one-hot crearia demasiadas columnas.

**Target encoding.** Reemplaza una categoria por la tasa historica de fraude de esa categoria. Puede ser potente, pero es peligroso porque usa la variable objetivo. Si se aplica, debe hacerse con folds temporales o solo con pasado; en esta primera etapa no sera la tecnica principal.

**Adversarial validation.** Entrena un modelo para distinguir train de test o una ventana de otra. Si logra separarlas facilmente, significa que las distribuciones cambiaron. No dice por si solo que el modelo de fraude sera malo, pero alerta que debemos validar temporalmente y monitorear drift.

**KS/PSI.** Son medidas de cambio de distribucion. Sirven para responder: "esta variable se comporta parecido en esta ventana y en la anterior?". No reemplazan las metricas supervisadas, porque una variable puede cambiar sin afectar performance, pero ayudan a explicar degradacion.

**Permutation importance.** Mide cuanto cae el desempeno cuando se desordena una feature. Si al desordenar una columna el modelo empeora mucho, esa columna era importante. Se usa como herramienta de interpretacion, no como prueba causal.

**Sliding window.** Reentrena usando solo datos recientes. Es util si los patrones viejos dejan de representar el presente.

**Expanding window.** Reentrena acumulando todo el historial disponible. Es util si el pasado sigue aportando informacion y se quiere aprovechar mas datos.

**Calibracion.** Ajusta las probabilidades para que sean mas interpretables. Si el modelo dice 0.20, idealmente transacciones parecidas deberian tener alrededor de 20% de fraude. Esto importa para la politica de decision, porque los umbrales se basan en costo esperado.

### Decision 1: comenzar con un baseline temporal

El primer notebook de modelado sera un baseline temporal con LightGBM.

Incluira:

- Merge transaction + identity.
- Reduccion de memoria.
- Features base del EDA.
- Split temporal.
- Metricas globales y por ventana.
- Reporte de PR-AUC, Recall, Precision, F1, ROC-AUC y FPR.

Razon:

LightGBM se elige como primer baseline porque el dataset es tabular, grande y mixto: tiene variables numericas, categoricas codificadas, faltantes y relaciones no lineales. Un modelo lineal simple exige mas supuestos sobre la forma de la relacion entre monto, tarjeta, identidad y fraude. En cambio, un modelo de boosting con arboles puede capturar interacciones como "cierto producto + cierto tipo de tarjeta + monto alto + ausencia de identity" sin que tengamos que escribir manualmente cada regla.

Esta decision no significa que LightGBM sea automaticamente el modelo final. Se usa primero porque permite obtener una referencia fuerte y relativamente rapida. Despues se comparara contra CatBoost, XGBoost y un baseline interpretable.

### Decision 2: no usar validacion aleatoria

La validacion sera temporal.

Razon:

El proyecto exige evaluar sobre datos futuros y el EDA mostro variacion temporal real. Un split aleatorio mezclaria periodos y daria una metrica demasiado optimista.

Ejemplo simple: si una modalidad de fraude aparece en semanas tardias, un split aleatorio podria poner parte de esas transacciones en entrenamiento y parte en validacion. El modelo pareceria detectar bien ese patron, pero en produccion el sistema habria tenido que enfrentarlo por primera vez sin ejemplos futuros. La validacion temporal evita esa trampa.

### Decision 3: reportar segmentos UID conocido y UID desconocido

Cada validacion temporal se separara en:

- todas las transacciones;
- transacciones con UID visto antes;
- transacciones con UID nuevo.

Razon:

Los UIDs conocidos y desconocidos tienen tasas de fraude diferentes y niveles de dificultad distintos. Reportar solo una metrica global ocultaria esa diferencia.

Esto tambien ayuda a interpretar errores. Si el modelo funciona bien en UID conocido pero mal en UID desconocido, entonces esta aprovechando memoria historica pero no generaliza bien a clientes nuevos. Si funciona razonablemente en ambos, la evidencia de robustez es mayor.

### Decision 4: crear features UID con control temporal

Se construiran UIDs candidatos y agregaciones historicas, pero calculadas solo con informacion disponible antes de la ventana de validacion.

Razon:

El EDA encontro que 42.76% de las filas tardias comparten UID candidato con el entrenamiento temprano y que la tasa de fraude es distinta entre UID conocido y UID desconocido. Esto indica que el historial de entidad puede contener senal predictiva relevante. Al mismo tiempo, esa misma senal puede generar leakage si se calcula con informacion futura. Por eso se usaran features UID, pero solo bajo una regla estricta de temporalidad.

La regla practica sera: para cada ventana de validacion, las frecuencias y agregaciones se ajustan usando solo la ventana de entrenamiento. Si una categoria o UID aparece por primera vez en validacion, se marca como desconocido y recibe valores por defecto definidos desde train, no desde validacion.

### Decision 5: tratar faltantes como informacion

Se agregaran indicadores de ausencia:

- `has_identity`;
- flags de bloques `V`;
- flags de columnas `D` e `id_*` relevantes.

Razon:

La ausencia de datos parece estructural, no aleatoria. Puede reflejar producto, canal o proceso de autenticacion.

Por ejemplo, si una transaccion no tiene bloque identity, eso no significa simplemente "dato perdido"; puede indicar que esa transaccion paso por un flujo donde no se recolecta identidad. Esa diferencia de proceso puede estar relacionada con riesgo. Por eso se crea `has_identity` y no se imputa todo como si la ausencia fuera accidental.

### Decision 6: conservar outliers de monto

No se eliminaran montos extremos.

Se agregaran:

- `log1p(TransactionAmt)`;
- flag de outlier;
- diferencia contra promedio historico de UID.

Razon:

Los outliers de monto tienen tasa de fraude superior al promedio y pueden representar senal relevante.

El uso de `log1p(TransactionAmt)` se justifica por la forma de la distribucion: el monto tiene una cola derecha muy larga. La transformacion logaritmica comprime diferencias enormes entre montos grandes, por ejemplo entre 1000 y 10000, sin perder el orden. Esto puede ayudar a modelos o componentes que son sensibles a escala. Sin embargo, como los arboles pueden manejar cortes sobre el monto original, no se reemplazara el monto crudo; se agregara `log1p` como feature candidata y se validara por ablation.

### Decision 7: dejar la politica de decision para despues del baseline predictivo

Primero se construira el modelo probabilistico. Luego se definiran umbrales de decision:

- aprobar;
- revisar;
- escalar.

Razon:

La politica de decision necesita probabilidades razonablemente calibradas y una validacion estable. No conviene elegir umbrales antes de tener un modelo base.

La razon practica es esta: si todavia no sabemos si los scores del modelo ordenan bien los fraudes, elegir umbrales seria prematuro. Primero se evalua si el modelo separa bien riesgo alto y bajo. Despues se decide que accion tomar para cada nivel de riesgo, considerando costos y capacidad de revision.

### Decision 8: no aplicar balanceo artificial como primera opcion

En esta primera etapa no se usara sobremuestreo agresivo de fraude ni submuestreo fuerte de transacciones legitimas como estrategia principal.

Razon:

El desbalance es real y forma parte del problema operacional. Si alteramos demasiado la proporcion de fraude durante el entrenamiento, el modelo puede aprender probabilidades mal calibradas o una frontera que no representa la frecuencia real del evento. Primero se usaran metricas adecuadas al desbalance y pesos de clase o parametros del modelo si hacen falta. Tecnicas de resampling quedaran como experimento posterior, no como punto de partida.

### Decision 9: cada feature nueva debe pasar por ablation

Toda familia de features se probara comparando el modelo con y sin esa familia:

- base;
- base + monto transformado;
- base + missing indicators;
- base + UID features;
- base + UID + agregaciones historicas.

Razon:

Esto evita justificar una tecnica solo por intuicion. Si una feature suena razonable pero no mejora validacion temporal, no queda en el modelo recomendado. La evidencia final debe salir de comparaciones controladas.

## 4. Plan de notebooks

### Notebook 01 - Baseline temporal

Objetivo:

Construir el primer modelo predictivo defendible.

Contenido:

- carga de datos;
- merge;
- features base;
- split temporal;
- LightGBM;
- metricas globales y temporales.

Output:

- `baseline_metrics.csv`;
- `baseline_predictions_valid.csv`;
- `baseline_feature_importance.csv`;
- `baseline_decision_log.md`.

### Notebook 02 - Features UID

Objetivo:

Incorporar informacion historica de cliente/tarjeta.

Contenido:

- `D1n`, `D2n`, `D10n`, `D15n`;
- UIDs candidatos;
- frequency encoding;
- agregaciones historicas por UID;
- evaluacion por UID conocido/desconocido.

Output:

- comparacion baseline vs UID;
- metricas por segmento;
- importancia de features UID.

### Notebook 03 - Auditoria de features

Objetivo:

Identificar features inestables, redundantes o riesgosas.

Contenido:

- KS train/test;
- adversarial validation;
- permutation importance;
- estabilidad temporal;
- correlaciones entre columnas C/V/D.

Justificacion:

Este notebook responde una pregunta distinta al modelado: no pregunta "que modelo da mas score?", sino "en que variables podemos confiar?". Esto importa porque una variable puede parecer predictiva en train, pero cambiar mucho en el tiempo o entre train/test. Si el modelo depende demasiado de una variable inestable, su desempeno puede caer en ventanas futuras.

El adversarial validation se usa para detectar si una ventana puede distinguirse de otra. El KS/PSI se usa para medir que variables cambiaron. La permutation importance se usa para saber si el modelo esta dependiendo de esas variables. La combinacion de las tres herramientas permite distinguir:

- feature predictiva y estable: candidata a conservar;
- feature predictiva pero inestable: conservar con monitoreo o penalizar;
- feature no predictiva y redundante: candidata a eliminar;
- feature que separa train/test pero no fraude: riesgo de sobreajuste temporal.

Output:

- `feature_audit.csv`;
- lista de features seguras;
- lista de features a remover o monitorear.

### Notebook 04 - Comparacion de modelos

Objetivo:

Comparar algoritmos bajo el mismo protocolo.

Modelos:

- LightGBM;
- CatBoost;
- XGBoost;
- regresion logistica como baseline interpretable.

Justificacion:

La comparacion no busca "probar muchos modelos porque si". Cada modelo cumple un rol:

- Regresion logistica: baseline simple e interpretable. Sirve para saber cuanto se puede lograr con relaciones casi lineales y transformaciones basicas.
- LightGBM: baseline fuerte para datos tabulares grandes, faltantes e interacciones no lineales.
- CatBoost: candidato util si las categoricas aportan mucha senal y queremos tratarlas de forma mas natural.
- XGBoost: alternativa de boosting con comportamiento distinto, util para verificar que los resultados no dependan de una sola implementacion.

La seleccion final no sera "el de mayor metrica global" automaticamente. Se elegira considerando PR-AUC, recall, FPR, estabilidad temporal, desempeno en UID desconocido y costo de inferencia.

Output:

- `model_comparison.csv`;
- metricas por ventana;
- metricas por UID conocido/desconocido;
- tiempos de entrenamiento e inferencia.

### Notebook 05 - Politica de decision

Objetivo:

Transformar score de fraude en accion.

Contenido:

- funcion de costo;
- capacidad maxima de revision;
- busqueda de umbrales;
- matriz de decisiones.

Output:

- `policy_thresholds.csv`;
- curva costo vs capacidad;
- reporte de fraude no detectado y carga de revision.

Justificacion:

Un modelo de fraude no se usa para publicar una probabilidad y terminar. Se usa para decidir. Si el score es bajo, la transaccion podria aprobarse automaticamente. Si es intermedio, podria ir a revision. Si es muy alto, podria escalarse o bloquearse.

La politica debe considerar que los errores no cuestan lo mismo:

- Falso negativo: fraude que pasa sin detectar.
- Falso positivo: compra legitima marcada como sospechosa.
- Revision manual: costo operativo y capacidad limitada.

Por eso los umbrales no se eligen por intuicion. Se seleccionan en validacion minimizando costo esperado y respetando una capacidad maxima de revision.

### Notebook 06 - Monitoreo de drift

Objetivo:

Detectar degradacion temporal.

Contenido:

- PR-AUC por ventana;
- Recall por ventana;
- F1 por ventana;
- KS/PSI de variables clave;
- tasa de fraude por ventana;
- tasa de revision por ventana.

Output:

- `drift_report.csv`;
- graficos de degradacion.

Justificacion:

El EDA ya mostro que la tasa de fraude no es estable. Por eso el modelo debe monitorearse por ventanas. No basta con saber el PR-AUC promedio: si el promedio es bueno pero hay semanas donde cae mucho el recall, el sistema seria riesgoso.

Se monitorean dos cosas juntas:

- cambios en datos, mediante KS/PSI y tasas por variable;
- cambios en desempeno, mediante PR-AUC, recall y F1 por ventana.

Solo cuando ambos aparecen juntos se puede argumentar que un cambio de distribucion esta asociado a degradacion predictiva.

### Notebook 07 - Adaptacion

Objetivo:

Comparar modelo estatico vs modelo adaptativo.

Estrategias:

- modelo estatico;
- expanding window;
- sliding window;
- reentrenamiento condicionado por drift.

Output:

- comparacion static vs adaptive;
- recomendacion de estrategia final.

Justificacion:

El proyecto pide adaptacion, pero adaptarse no significa reentrenar todo el tiempo. Reentrenar tiene costo y tambien puede hacer que el modelo olvide patrones utiles. Por eso se comparan estrategias:

- Modelo estatico: sirve como referencia. Si no se degrada, quizas no hace falta adaptar.
- Expanding window: conserva todo el historial y agrega datos nuevos.
- Sliding window: prioriza datos recientes cuando el pasado se vuelve menos representativo.
- Reentrenamiento condicionado: actualiza solo si hay senales de drift o caida de metricas.

La estrategia recomendada sera la que reduzca degradacion temporal sin aumentar innecesariamente costo o inestabilidad.

## 5. Criterios de exito de esta primera etapa

La primera etapa se considerara completa cuando exista:

- Un baseline temporal reproducible en Kaggle.
- Una tabla de metricas con PR-AUC, Recall, Precision, F1, ROC-AUC y FPR.
- Separacion de resultados global, UID conocido y UID desconocido.
- Un primer decision log que explique cada decision tomada.
- Evidencia de que no se uso informacion futura en features o encodings.

## 6. Riesgos metodologicos y mitigaciones

### Riesgo: leakage temporal

Mitigacion:

- No usar split aleatorio.
- No calcular encodings con validacion incluida.
- No usar `TransactionDT` crudo como predictor.

### Riesgo: sobreajuste a UIDs conocidos

Mitigacion:

- Reportar UID conocido y desconocido por separado.
- Validar UIDs en ventanas futuras.

### Riesgo: perder senal al eliminar features

Mitigacion:

- No eliminar agresivamente en baseline.
- Auditar antes de remover.

### Riesgo: metrica optimista por desbalance

Mitigacion:

- Priorizar PR-AUC.
- Reportar Recall, Precision y F1.
- Evitar accuracy como criterio de exito.

### Riesgo: modelo bueno pero decision mala

Mitigacion:

- En etapa posterior, seleccionar umbrales segun costo esperado y capacidad de revision.

## 7. Referencias

### Sustento metodologico principal

- Bolton, R. J., & Hand, D. J. (2002). Statistical Fraud Detection: A Review. Statistical Science. https://doi.org/10.1214/ss/1042727940
- Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. ACM Computing Surveys. https://doi.org/10.1145/2523813
- Krawczyk, B. (2016). Learning from imbalanced data: open challenges and future directions. Progress in Artificial Intelligence. https://doi.org/10.1007/s13748-016-0094-0
- Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE. https://doi.org/10.1371/journal.pone.0118432
- scikit-learn documentation. TimeSeriesSplit. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

### Antecedentes tecnicos consultados, no usados como justificacion principal

- Kaggle IEEE-CIS Fraud Detection, 1st place solution. https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
- Kaggle IEEE-CIS Fraud Detection, 2nd place solution. https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/2-uncles-and-3-puppies-2nd-solution-cpmp-view
