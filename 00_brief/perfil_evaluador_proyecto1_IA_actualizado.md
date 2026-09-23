# Perfil actualizado del evaluador --- Proyecto 1: Sistema Inteligente Adaptativo con Datos No Estacionarios

## Propósito

Este documento actualiza el perfil preliminar del evaluador inferido a
partir de la Hackathon 1 y lo adapta específicamente al marco del
**Proyecto 1: Sistema Inteligente Adaptativo con Datos No
Estacionarios**.

Se deben separar dos fuentes de evidencia:

1.  **Patrones observados del evaluador:** derivados de la calificación
    y retroalimentación previa.
2.  **Exigencias explícitas del Proyecto 1:** derivadas del enunciado y
    su rúbrica.

Las exigencias del proyecto tienen prioridad. El perfil previo funciona
como un **prior** para anticipar cómo puede interpretar el profesor la
evidencia presentada, no como sustituto de la rúbrica.

------------------------------------------------------------------------

## Actualización a partir del enunciado/material más reciente

### Nueva evidencia de alta prioridad

El material actualizado introduce señales concretas que deben modificar
la forma de evaluar y priorizar el proyecto:

-   **Drift:** se favorece una estrategia sencilla y operacionalizable.
    Para este proyecto, *sliding window* aparece como una alternativa
    natural de adaptación.
-   **Comparación metodológica:** debe existir una comparación explícita
    entre varios métodos; la nota incluida en el material indica
    **cuatro métodos** como referencia de trabajo.
-   **Granularidad temporal:** el análisis debe organizarse **por
    meses**, no únicamente mediante un split cronológico global.
-   **Política de calidad:** el sistema necesita **thresholds
    explícitos** que determinen cuándo el desempeño deja de ser
    aceptable.
-   **Regla de adaptación:** si el modelo **no cumple la política de
    calidad**, debe activarse el reentrenamiento.
-   **Análisis de escenarios:** la evaluación debe incluir escenarios
    que permitan observar cómo responde el sistema bajo distintas
    condiciones temporales y operacionales.

Estas señales elevan la importancia de una arquitectura con una regla de
control explícita:

> **medir por periodo → comparar con thresholds de calidad → si cumple,
> mantener modelo → si no cumple, reentrenar → validar candidato →
> promover solo si recupera calidad.**

Esto es más concreto que una formulación genérica de "detectar drift y
adaptarse": el sistema debe demostrar **cuándo considera que existe un
problema operativo y qué acción ejecuta en consecuencia**.

### Recalibración del criterio de excelencia

Para aspirar al nivel máximo, ya no basta con mostrar que las
distribuciones cambian o que un modelo adaptativo obtiene mejores
métricas en promedio. Debe quedar trazable:

**mes → métricas observadas → política/threshold de calidad → decisión
de mantener o reentrenar → evaluación posterior.**

Por tanto, una entrega fuerte debe mostrar al menos:

1.  evaluación temporal mensual;
2.  política de calidad cuantificada;
3.  comparación de métodos bajo el mismo protocolo;
4.  regla explícita de reentrenamiento;
5.  análisis de escenarios;
6.  evidencia de qué ocurre después de adaptar el modelo.

### Riesgo principal actualizado

El mayor riesgo metodológico pasa a ser implementar componentes
correctos de manera aislada, pero **sin una política operacional que los
conecte**.

Ejemplo débil:

> "Detectamos data drift con PSI y usamos sliding window."

Ejemplo fuerte:

> "Evaluamos el sistema mensualmente. Si una métrica crítica cae por
> debajo del threshold definido en validación, el modelo incumple la
> política de calidad y se activa un reentrenamiento con una ventana
> temporal reciente. El candidato se valida temporalmente y solo
> reemplaza al vigente si recupera los criterios de calidad."

------------------------------------------------------------------------

## 1. Regla central de evaluación

No basta con que una decisión sea técnicamente razonable. Para maximizar
puntaje debe ser:

> **explícita + específica al caso de uso + justificada + medible +
> conectada con el comportamiento temporal del sistema.**

El patrón observado en Hackathon 1 sugiere poco crédito para contenido
implícito. En este proyecto el riesgo aumenta porque la rúbrica exige no
solo componentes, sino también **justificación, análisis temporal y
coherencia del ciclo de vida completo**.

### Test operativo

Para cada criterio debe poder señalarse inmediatamente:

1.  dónde está la evidencia;
2.  qué decisión concreta tomó el equipo;
3.  por qué se tomó;
4.  cómo se evaluó;
5.  cómo cambia o se mantiene válida cuando los datos evolucionan.

Si alguno requiere inferencia importante, existe riesgo de perder el
nivel **Excelente (4)**.

------------------------------------------------------------------------

## 2. Cambio de marco: de Hackathon a Proyecto 1

La Hackathon privilegiaba una propuesta clara y defendible. El Proyecto
1 exige además **evidencia experimental y trazabilidad end-to-end**.

Por tanto, la regla anterior:

> "Una solución simple, específica, coherente y defendible vale más que
> una arquitectura sofisticada pero dispersa"

se mantiene, pero se amplía a:

> **Una solución simple, específica, reproducible y temporalmente
> validada vale más que una solución sofisticada sin evidencia de cómo
> se comporta y adapta ante drift.**

El proyecto no debe parecer una colección de técnicas. Debe contar una
única historia:

**problema real → datos temporales → predicción → decisión → acción →
monitoreo → detección de drift → adaptación → nueva evaluación.**

------------------------------------------------------------------------

## 3. Modelo de puntuación recalibrado: 4 / 3 / 2 / 0--1

### Excelente --- 4

-   El criterio está cubierto explícitamente y de forma completa.
-   Las decisiones son específicas al caso de uso.
-   Existe justificación técnica u operacional.
-   La evidencia es cuantitativa cuando corresponde.
-   Se respeta la temporalidad de los datos.
-   La decisión es coherente con el resto del sistema.
-   Se puede defender sin depender de explicaciones adicionales
    importantes.

### Bueno --- 3

-   El criterio está correctamente cubierto.
-   Existe evidencia relevante.
-   Falta profundidad, justificación, comparación, cuantificación o
    análisis temporal en alguna parte.

### Básico --- 2

-   Existe una implementación o explicación funcional.
-   La cobertura es parcial, genérica o débilmente conectada con la
    naturaleza temporal del problema.
-   Varias decisiones aparecen sin suficiente evidencia o justificación.

### Insuficiente --- 0--1

-   El criterio es confuso, incompleto o incorrecto.
-   Se ignora un requisito central.
-   Se viola la temporalidad de los datos.
-   El evaluador tendría que reconstruir o inferir la respuesta.

### Regla conservadora

No subir una sección a 4/4 solo porque "todo está mencionado".
**Excelente exige cierre:** decisión + justificación + evidencia +
interpretación.

------------------------------------------------------------------------

## 4. Qué parece valorar especialmente el evaluador

### 4.1 Explícito \> implícito

Cada elemento solicitado debe ser localizable sin reconstruir la
intención del equipo.

En Proyecto 1 esto aplica especialmente a:

-   problema de clasificación;
-   usuario o actor que consume la predicción;
-   decisión que habilita el sistema;
-   restricciones;
-   métricas técnicas, de decisión y sociales;
-   tratamiento de incertidumbre;
-   acción generada;
-   mecanismo de detección de drift;
-   regla de adaptación.

### 4.2 Concreción \> sofisticación

No agregar modelos, módulos o técnicas porque "suenan avanzados".

Para cada componente:

> **¿Qué problema concreto resuelve y qué evidencia mostraría que
> funciona?**

Si no hay una respuesta clara, probablemente sobra.

### 4.3 Trazabilidad directa

El documento debe permitir seguir:

**objetivo → riesgo/costo de error → métrica → modelo → evaluación
temporal → decisión → monitoreo → adaptación.**

Una métrica aislada o un detector de drift sin consecuencia operacional
tiene menor valor.

### 4.4 Cierre de decisiones

El evaluador previo penalizó temas abiertos. En este proyecto evitar
frases como:

-   "se podría usar..."
-   "consideraríamos reentrenar..."
-   "una opción sería..."

cuando el proyecto ya permite tomar una decisión.

Preferir:

> "Usamos ADWIN para detectar cambios; ante una alerta sostenida durante
> X ventanas, reentrenamos con las últimas N semanas y promovemos el
> modelo solo si supera al vigente bajo el protocolo temporal definido."

Los valores exactos deben justificarse o declararse como parámetros
experimentales, no inventarse.

------------------------------------------------------------------------

## 5. Criterio 1 --- Problema, objetivos, restricciones y métricas

### Para aspirar a 4/4

Debe quedar explícito:

-   **caso de uso real**;
-   **tarea de clasificación supervisada**;
-   **qué predice** el modelo;
-   **para quién**;
-   **qué decisión** mejora;
-   **horizonte temporal** relevante;
-   **restricciones** de datos, operación y recursos;
-   **acciones que el sistema no debe tomar autónomamente**, cuando
    corresponda;
-   métricas **técnicas**;
-   métricas de **decisión**;
-   métricas **sociales**;
-   justificación basada en el costo de error.

### Cadena recomendada

**Objetivo → error relevante → métrica → target/baseline → impacto
decisional.**

Ejemplo conceptual:

> Detectar transacciones fraudulentas antes de autorización → un falso
> negativo permite fraude y un falso positivo bloquea una operación
> legítima → priorizar recall sin ignorar precision/costo → evaluar
> además costo esperado por ventana temporal.

### Red flags

-   "predecir fraude con alta precisión" como objetivo completo;
-   listar accuracy/F1/AUC sin explicar cuál gobierna la decisión;
-   métricas sociales decorativas;
-   targets sin baseline o fundamento;
-   restricciones que en realidad son fallos;
-   no explicar qué ocurre después de la clasificación.

------------------------------------------------------------------------

## 6. Criterio 2 --- Análisis y preparación de datos

Este criterio debe tratarse como **análisis temporal**, no como EDA
convencional con una columna de fecha.

### Para 4/4

Buscar evidencia de:

1.  evolución temporal de variables y target;
2.  cambios de distribución;
3.  anomalías;
4.  calidad y faltantes a través del tiempo;
5.  posible cambio en prevalencia de clases;
6.  features temporales justificadas;
7.  split estrictamente cronológico;
8.  prevención explícita de leakage.

### Evidencia visual de alto ROI

-   prevalencia de clase por periodo;
-   distribución de variables importantes en distintas ventanas;
-   missingness por periodo;
-   métricas de distancia o tests entre ventanas;
-   línea temporal del split train/validation/test.

### Regla crítica

Un **random split** contradice directamente el enfoque del proyecto. La
evaluación debe simular:

> **entrenar con pasado → predecir futuro.**

### Features temporales

Ventanas deslizantes, lags y agregaciones solo suman si respetan
causalidad:

> para predecir en tiempo (t), una feature no puede usar información
> posterior a (t).

------------------------------------------------------------------------

## 7. Criterio 3 --- Modelado y evaluación temporal

El enunciado exige **al menos dos enfoques tradicionales y uno más
avanzado**.

### Para 4/4

Debe existir:

-   baseline claro;
-   dos modelos tradicionales;
-   un modelo avanzado;
-   justificación de cada familia;
-   protocolo comparable;
-   métricas adecuadas;
-   evaluación por periodos/ventanas;
-   análisis de degradación;
-   interpretación del comportamiento temporal.

### Lo que NO basta

Una tabla global:

Modelo F1 AUC -------- ---- -----

puede demostrar rendimiento promedio, pero no demuestra robustez
temporal.

### Evidencia preferida

Añadir:

-   rendimiento por ventana temporal;
-   variabilidad entre periodos;
-   caída respecto a una ventana de referencia;
-   relación entre drift detectado y degradación;
-   comparación contra baseline.

### Pregunta que debe poder responder el equipo

> **¿El mejor modelo promedio sigue siendo el mejor cuando cambia la
> distribución?**

Si no se puede responder, el análisis temporal está incompleto.

------------------------------------------------------------------------

## 8. Criterio 4 --- Diseño del sistema y estrategia de adaptación

Este criterio concentra gran parte de la esencia del curso.

### Arquitectura mínima esperada

**fuentes de datos → validación/preprocesamiento → features → modelo
predictivo → incertidumbre/confianza → regla de decisión → acción →
feedback/labels → monitoreo → detector de drift →
adaptación/reentrenamiento.**

El enunciado también solicita considerar **fuentes multimodales**. Si el
dataset elegido no las ofrece realmente, no inventar una implementación
multimodal: distinguir con claridad entre los datos implementados y
posibles fuentes adicionales en un despliegue real.

### Para 4/4

El diseño debe incluir explícitamente:

-   adquisición e integración;
-   módulo predictivo;
-   toma de decisiones;
-   incertidumbre;
-   acción;
-   flujo de datos;
-   arquitectura de implementación;
-   frecuencia de actualización;
-   autonomía;
-   escalabilidad/costos/integración;
-   detección de drift;
-   mecanismo de adaptación.

### Test de sistema, no solo de modelo

Debe quedar claro:

> **¿Qué ocurre desde que llega un dato hasta que alguien o algo toma
> una acción?**

y luego:

> **¿Qué ocurre cuando el sistema deja de ser confiable?**

------------------------------------------------------------------------

## 9. Concept drift: estándar de evaluación elevado

Concept drift no debe aparecer como una sección añadida al final. Es el
**eje transversal** del proyecto.

### Cadena mínima

**señal de cambio → detector → criterio de alerta → impacto observado →
respuesta → validación post-adaptación.**

### Distinguir explícitamente

-   **Data/covariate drift:** cambia (P(X)).
-   **Label/prior drift:** cambia (P(Y)).
-   **Concept drift:** cambia la relación relevante entre entradas y
    salida, típicamente (P(Y\|X)).

No afirmar concept drift únicamente porque cambió una feature. Si no
puede demostrarse directamente, usar lenguaje preciso: "señal de
drift/distribution shift" y relacionarla con degradación predictiva.

### Detector sin política de acción = incompleto

No basta con "usamos PSI/KS/ADWIN".

Debe responderse:

> ¿Qué hace el sistema cuando el detector dispara?

### Adaptación sin protocolo de promoción = incompleta

Después de reentrenar:

-   evaluar en un bloque temporal válido;
-   comparar con el modelo vigente;
-   definir criterio de promoción/rollback;
-   registrar versión y periodo de entrenamiento.

------------------------------------------------------------------------

## 10. Incertidumbre y autonomía

El enunciado exige manejo de incertidumbre y análisis del nivel de
autonomía.

No tratar "probabilidad del clasificador" automáticamente como
incertidumbre bien calibrada.

### Preguntas esperables

-   ¿La confianza está calibrada?
-   ¿Qué pasa cerca del umbral?
-   ¿Cuándo interviene un humano?
-   ¿Qué decisiones están prohibidas al sistema?
-   ¿Qué ocurre ante datos fuera de distribución?
-   ¿El drift cambia el nivel de confianza permitido?

### Patrón fuerte

**predicción + confianza + contexto → regla de decisión → acción o
escalamiento humano.**

------------------------------------------------------------------------

## 11. Producto entregado: 5 páginas implica compresión estratégica

El informe técnico tiene un máximo de cinco páginas. Por ello, la
exhaustividad no significa escribir más, sino seleccionar evidencia de
alto valor.

### Prioridad visual

1.  arquitectura end-to-end;
2.  timeline del split/protocolo temporal;
3.  tabla compacta de modelos;
4.  gráfico de rendimiento en el tiempo;
5.  gráfico/señal de drift alineado temporalmente con rendimiento;
6.  protocolo de adaptación;
7.  infografía final del ciclo de vida.

Una figura que demuestra dos relaciones importantes vale más que varias
visualizaciones de EDA genérico.

### Código

La reproducibilidad debe permitir reconstruir:

**preprocesamiento → entrenamiento → evaluación temporal → detección →
adaptación.**

Evitar notebooks cuyo resultado dependa de ejecución manual fuera de
orden.

------------------------------------------------------------------------

## 12. Infografía final

No debe ser una arquitectura duplicada.

Debe sintetizar el **ciclo de vida dinámico**:

**datos → entrenamiento → despliegue → predicción/decisión → monitoreo →
drift → adaptación → validación → nuevo despliegue.**

Su función es demostrar que el equipo entiende el sistema como un
proceso continuo, no como un modelo entrenado una sola vez.

------------------------------------------------------------------------

## 13. Matriz de trazabilidad recomendada

  -------------------------------------------------------------------------
  Criterio              Evidencia mínima para 4/4     Riesgo principal
  --------------------- ----------------------------- ---------------------
  Problema, objetivos,  Caso de uso + decisión +      Métricas genéricas o
  restricciones y       restricciones + 3 tipos de    desconectadas
  métricas              métricas justificadas         

  Análisis y            EDA temporal +                EDA estático o
  preparación           shifts/anomalías + features   leakage
                        causales + split cronológico  

  Modelado y evaluación 2 tradicionales + 1           Solo promedio global
  temporal              avanzado + baseline +         
                        métricas por tiempo           

  Diseño y adaptación   Arquitectura completa +       Detector sin acción /
                        incertidumbre + acción +      arquitectura
                        drift + política de           sobrediseñada
                        adaptación                    

  Calidad del producto  Pipeline reproducible +       Documento completo
                        figuras/tablas/protocolos +   pero poco integrado
                        infografía                    
  -------------------------------------------------------------------------

------------------------------------------------------------------------

## 14. Preguntas probables del evaluador

### Problema y métricas

-   ¿Qué decisión concreta cambia gracias a la predicción?
-   ¿Por qué esta métrica y no accuracy?
-   ¿Cuál es el costo de falso positivo y falso negativo?
-   ¿Cómo se mide realmente la métrica social?

### Datos

-   ¿Por qué ese corte temporal?
-   ¿Hay leakage en estas features?
-   ¿Cómo cambia la prevalencia de clases?
-   ¿Qué evidencia muestra que la distribución cambió?

### Modelos

-   ¿Por qué el modelo avanzado es necesario?
-   ¿Contra qué baseline se compara?
-   ¿Cuál modelo es más estable en el tiempo?
-   ¿Una mejora promedio oculta periodos malos?

### Drift y adaptación

-   ¿Cómo distinguen drift de ruido?
-   ¿Qué dispara el reentrenamiento?
-   ¿Qué datos entran al nuevo entrenamiento?
-   ¿Cómo validan el nuevo modelo sin usar futuro?
-   ¿Qué pasa si el modelo reentrenado es peor?
-   ¿Cómo evitan reaccionar demasiado a cambios transitorios?

### Sistema

-   ¿Qué hace el sistema ante baja confianza?
-   ¿Qué hace automáticamente y qué requiere humano?
-   ¿Con qué frecuencia se monitorea y por qué?
-   ¿Cuál es el costo operativo de reentrenar?
-   ¿Qué parte implementaron y qué parte es propuesta de despliegue?

------------------------------------------------------------------------

## 15. Priorización del tiempo

### Prioridad 1 --- Protocolo temporal correcto

Antes de optimizar modelos, cerrar split, ventanas, causalidad y
protocolo de evaluación.

### Prioridad 2 --- Trazabilidad problema → decisión → métricas

Si el objetivo y el costo de error no están claros, las métricas y
umbrales tampoco lo estarán.

### Prioridad 3 --- Evidencia real de degradación/drift

Construir gráficos que muestren qué cambia, cuándo cambia y qué efecto
tiene.

### Prioridad 4 --- Adaptación evaluable

Definir trigger, datos de reentrenamiento, validación y criterio de
promoción.

### Prioridad 5 --- Arquitectura end-to-end simple

Mostrar el sistema completo sin módulos decorativos.

### Prioridad 6 --- Comparación de modelos

Solo después de tener un protocolo temporal válido.

### Prioridad 7 --- Presentación visual

Comprimir la evidencia para las cinco páginas y la infografía.

------------------------------------------------------------------------

## 16. Procedimiento para evaluar futuras entregas del Proyecto 1

### Paso 1 --- Construir mapa rúbrica → evidencia

Para cada uno de los cinco criterios, identificar páginas, figuras,
tablas, código o experimentos que lo demuestran.

### Paso 2 --- Buscar violaciones temporales primero

Antes de discutir modelos:

-   random split;
-   leakage;
-   features que usan futuro;
-   tuning sobre test;
-   comparación de ventanas inconsistente.

Una violación aquí puede invalidar gran parte de la evidencia posterior.

### Paso 3 --- Evaluar cada criterio sin inferencias favorables

Solo contar lo que el producto realmente demuestra.

### Paso 4 --- Asignar 4 / 3 / 2 / 0--1

Explicar qué evidencia falta para subir exactamente un nivel.

### Paso 5 --- Revisar coherencia transversal

Comprobar:

-   objetivo ↔ métricas;
-   métricas ↔ decisión;
-   datos ↔ features;
-   drift ↔ degradación;
-   detector ↔ adaptación;
-   adaptación ↔ validación;
-   arquitectura ↔ implementación.

### Paso 6 --- Priorizar cambios de máximo ROI

Favorecer cambios que:

-   eviten invalidación metodológica;
-   conviertan evidencia implícita en explícita;
-   añadan una justificación que cierre una decisión;
-   conviertan un resultado global en temporal;
-   conecten drift con una acción concreta;
-   permitan defender una pregunta probable.

------------------------------------------------------------------------

## 17. Formato recomendado para futuras revisiones

  ------------------------------------------------------------------------------
  Criterio         Nota estimada Confianza         Evidencia       Qué impide
                                                                   4/4
  --------------- -------------- ----------------- --------------- -------------
  Problema,                 0--4 Alta/Media/Baja   Página/Figura   ...
  objetivos,                                                       
  restricciones y                                                  
  métricas                                                         

  Análisis y                0--4 Alta/Media/Baja   Página/Figura   ...
  preparación de                                                   
  datos                                                            

  Modelado y                0--4 Alta/Media/Baja   Página/Figura   ...
  evaluación                                                       
  temporal                                                         

  Diseño del                0--4 Alta/Media/Baja   Página/Figura   ...
  sistema y                                                        
  adaptación                                                       

  Calidad del               0--4 Alta/Media/Baja   Página/Figura   ...
  producto                                                         
  ------------------------------------------------------------------------------

Para cada criterio:

**Nota estimada:** X/4\
**Evidencia:** qué demuestra realmente la entrega.\
**Cómo podría verlo el evaluador:** interpretación usando el perfil
observado.\
**Qué falta para 4/4:** cambio mínimo y concreto.\
**Riesgo metodológico:** si existe una decisión que pueda invalidar la
evidencia.

Al final:

-   riesgos de mayor pérdida potencial;
-   cambios de máximo ROI;
-   preguntas probables;
-   checklist de reproducibilidad y leakage.

------------------------------------------------------------------------

## 18. Nivel de confianza

### Alta confianza --- porque está explícito en el enunciado

-   naturaleza temporal obligatoria;
-   clasificación supervisada;
-   evaluación a lo largo del tiempo;
-   al menos dos modelos tradicionales y uno avanzado;
-   análisis de drift y degradación;
-   estrategia de adaptación;
-   sistema completo con predicción, decisión, incertidumbre y acción;
-   reproducibilidad;
-   informe máximo de cinco páginas;
-   diagramas, tablas, protocolos e infografía.

### Confianza media --- patrón observado del evaluador

-   preferencia por evidencia explícita;
-   separación semántica estricta;
-   preferencia por concreción sobre complejidad;
-   importancia de justificar decisiones;
-   penalización de temas abiertos;
-   sensibilidad a contenido generado por IA no auditado.

Estas últimas siguen siendo hipótesis derivadas principalmente de
Hackathon 1 y deben recalibrarse con la calificación real del Proyecto
1.

------------------------------------------------------------------------

## Checklist actualizado de máximo ROI

Antes de cerrar cualquier entrega, verificar:

-   [ ] El análisis y la evaluación están organizados temporalmente
    **por meses**.
-   [ ] Se comparan los métodos requeridos bajo exactamente el mismo
    protocolo temporal.
-   [ ] Existe una **política de calidad** con thresholds definidos y
    justificados.
-   [ ] Está escrito qué ocurre cuando el modelo **cumple** y cuando
    **no cumple** dicha política.
-   [ ] El reentrenamiento utiliza únicamente información disponible
    hasta ese momento.
-   [ ] La estrategia de *sliding window* especifica qué periodo
    reciente utiliza.
-   [ ] El modelo reentrenado no se despliega automáticamente: existe
    validación y criterio de promoción.
-   [ ] Se incluyen escenarios que ejerciten situaciones normales,
    degradación y adaptación.
-   [ ] Drift, degradación y fallo de política de calidad se distinguen
    semánticamente.
-   [ ] Cada gráfico o tabla conduce a una decisión o conclusión
    concreta.

### Preguntas adicionales que ahora deben poder responderse

-   ¿Por qué la evaluación temporal se realiza por meses?
-   ¿Cuáles son exactamente los thresholds de la política de calidad y
    cómo se fijaron?
-   ¿Qué métrica o combinación de métricas dispara el reentrenamiento?
-   ¿Qué ocurre si hay data drift pero el modelo sigue cumpliendo la
    política de calidad?
-   ¿Qué ocurre si no hay una señal fuerte de data drift pero el
    desempeño cae?
-   ¿Cuántos meses contiene la ventana de reentrenamiento y por qué?
-   ¿Qué cuatro métodos se comparan y qué dimensión de la solución
    representa cada uno?
-   ¿Qué escenarios se analizaron y qué decisión toma el sistema en cada
    uno?
-   ¿Cómo se demuestra que el reentrenamiento realmente recuperó
    calidad?
-   ¿Qué ocurre si el modelo candidato no supera al vigente?

## Resumen operativo

Para este proyecto, evaluar con tres preguntas sucesivas:

> **1. ¿Está explícitamente cubierto lo que pide la rúbrica?**

> **2. ¿La decisión está justificada y demostrada con evidencia temporal
> válida?**

> **3. ¿El sistema explica qué hace cuando el mundo cambia?**

La principal diferencia respecto al perfil de Hackathons es que ahora
**claridad y coherencia no bastan**: deben acompañarse de un protocolo
experimental temporal correcto, evidencia de degradación/drift y una
estrategia de adaptación concreta y evaluable.
