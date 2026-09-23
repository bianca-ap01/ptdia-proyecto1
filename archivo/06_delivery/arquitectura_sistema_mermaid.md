# Arquitectura del sistema

## Arquitectura end-to-end

```mermaid
flowchart LR
    A[Fuentes de datos transaccionales<br/>Transaction + Identity] --> B[Validacion e integracion<br/>schema checks, join, deduplicacion]
    B --> C[Feature pipeline temporal<br/>encoding, faltantes, monto, UID historico permitido]
    C --> D[Modelo predictivo V01<br/>LightGBM temporal]
    D --> E[Score de riesgo de fraude]
    E --> F[Calibracion opcional<br/>isotonic regression]
    E --> G[Politica V05<br/>umbrales approve/review/escalate]
    F --> G
    G --> H1[Aprobar<br/>bajo riesgo]
    G --> H2[Revision manual<br/>riesgo intermedio]
    G --> H3[Escalar o bloquear<br/>riesgo alto]
    H1 --> I[Logging de decision]
    H2 --> I
    H3 --> I
    I --> J[Monitoreo temporal<br/>PR-AUC, drift, costos, capacidad]
    J --> K{Alerta de drift<br/>o degradacion?}
    K -- No --> L[Continuar operacion]
    K -- Si --> M[Recalibrar umbrales<br/>o reentrenar con ventana reciente]
    M --> D
```

## Modulos exigidos por la rubrica

| Modulo | Funcion | Evidencia del proyecto |
| --- | --- | --- |
| Adquisicion e integracion | Recibe transacciones y atributos de identidad, une fuentes y valida estructura. | Kernels EDA/V01 integran transaction e identity. |
| Preparacion temporal | Ordena por `TransactionDT`, evita leakage y construye features permitidas. | V01 excluye tiempo absoluto como predictor. |
| Modulo predictivo | Estima score de fraude. | V01 LightGBM, holdout PR-AUC 0.5436. |
| Manejo de incertidumbre | Calibra score si se comunica como probabilidad. | V06 reduce Brier en holdout de 0.0479 a 0.0219. |
| Modulo de decision | Convierte score en accion. | V05: approve/review/escalate. |
| Componente de accion | Aprueba, envia a revision o escala. | Politica V05 reduce costo frente a reglas simples. |
| Monitoreo | Mide drift y degradacion temporal. | V03 feature audit, V06 sensibilidad. |
| Adaptacion | Recalibra, ajusta umbrales o reentrena. | Protocolo propuesto con ventanas deslizantes. |

Los modulos V05 y V06 estan publicados tambien como kernels Kaggle autocontenidos, por lo que la politica y la auditoria de robustez son reproducibles fuera del entorno local.

## Flujo de decision

```mermaid
flowchart TD
    A[Nueva transaccion] --> B[Generar features]
    B --> C[Score V01]
    C --> D{Score >= 0.779122?}
    D -- Si --> E[Escalar / bloquear<br/>riesgo extremo]
    D -- No --> F{Score >= 0.459783?}
    F -- Si --> G[Revision manual<br/>riesgo alto incierto]
    F -- No --> H[Aprobar automaticamente<br/>riesgo bajo]
    E --> I[Registrar resultado y feedback]
    G --> I
    H --> I
    I --> J[Monitoreo de drift y costos]
```

## Particion temporal

```mermaid
gantt
    title Particion temporal usada en modelado
    dateFormat  X
    axisFormat  %s
    section Dataset ordenado por TransactionDT
    Train 70%       :train, 0, 70
    Validacion 15%  :valid, 70, 15
    Holdout 15%     :holdout, 85, 15
```
