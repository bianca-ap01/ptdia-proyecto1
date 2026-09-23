# Infografia final en Mermaid

Esta infografia resume la entrega segun la rubrica del curso y conecta esa evidencia con el ciclo operativo del sistema adaptativo. La version final en LaTeX/PDF usa el mismo mensaje: problema, datos, modelado, sistema adaptativo, producto y presentacion.

```mermaid
flowchart TB
    A[Rubrica 1<br/>Problema, objetivos y metricas<br/>fraude 3.50%, PR-AUC, costo] --> B[Rubrica 2<br/>Analisis y preparacion<br/>EDA temporal, UID, faltantes, 70/15/15]
    B --> C[Rubrica 3<br/>Modelado y evaluacion<br/>V01-V07, holdout futuro, segmentos]
    C --> D[Rubrica 4<br/>Sistema adaptativo<br/>score, accion, incertidumbre, drift]
    D --> E[Rubrica 5<br/>Producto entregado<br/>GitHub, Kaggle, reportes, PDFs]
    E --> F[Rubrica 6<br/>Presentacion<br/>deck 15-20 min con defensa]
    D --> G[Ciclo operativo<br/>FastAPI + Docker + Cloud Run]
    G --> H[Monitoreo<br/>KS/PSI, Brier, costo, UID]
    H --> I{Alerta?}
    I -- No --> G
    I -- Si --> J[Adaptacion<br/>calibrar, ajustar umbrales o reentrenar]
    J --> C

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

La infografia debe ayudar al profesor a ver rapidamente que la entrega cubre la rubrica completa. El sistema aprende de datos historicos, decide bajo restricciones operativas, monitorea drift y actualiza umbrales o modelo cuando el entorno cambia. La trazabilidad queda respaldada por kernels publicos de Kaggle para EDA y V01-V07.
