# Infografia final en Mermaid

Esta infografia resume el ciclo de vida completo del sistema inteligente adaptativo.

```mermaid
flowchart TB
    A[1. Datos<br/>IEEE-CIS transaction + identity<br/>transacciones temporales] --> B[2. EDA temporal<br/>desbalance 3.50%<br/>drift semanal 1.85%-5.06%<br/>UID conocido/desconocido]
    B --> C[3. Preparacion<br/>split temporal 70/15/15<br/>sin TransactionDT crudo<br/>features tabulares y faltantes]
    C --> D[4. Modelado<br/>V01 LightGBM<br/>holdout PR-AUC 0.5436<br/>ROC-AUC 0.9051]
    D --> E[5. Decision<br/>V05 approve / review / escalate<br/>fraude detectado 66.17%<br/>revision 5.10%]
    E --> F[6. Despliegue<br/>FastAPI + Docker<br/>Artifact Registry<br/>Cloud Run]
    F --> G[7. Monitoreo<br/>PR-AUC por ventana<br/>KS/PSI<br/>score drift<br/>costos y capacidad]
    G --> H{8. Drift o degradacion?}
    H -- No --> F
    H -- Si --> I[9. Adaptacion<br/>recalibrar<br/>ajustar umbrales<br/>reentrenar con ventana deslizante]
    I --> D

    style A fill:#E8F3FF,stroke:#2B6CB0
    style B fill:#EAF8EA,stroke:#2F855A
    style C fill:#FFF7E6,stroke:#B7791F
    style D fill:#F0EBFF,stroke:#6B46C1
    style E fill:#FFECEC,stroke:#C53030
    style F fill:#E6FFFA,stroke:#2C7A7B
    style G fill:#F7FAFC,stroke:#4A5568
    style H fill:#FFFFFF,stroke:#1A202C
    style I fill:#FFF5F5,stroke:#C53030
```

## Mensaje para exposicion

El sistema no es un clasificador estatico. Es un ciclo adaptativo: aprende de datos historicos, decide bajo restricciones operativas, monitorea drift y actualiza umbrales o modelo cuando el entorno cambia.
