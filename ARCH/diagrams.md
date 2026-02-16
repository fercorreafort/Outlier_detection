## System Architecture (Example from your project)

```mermaid
graph TB
    subgraph "Data Sources"
        K[Kaggle API]
    end
    
    subgraph "Docker"
        PG[(PostgreSQL)]
        OS[(OpenSearch)]
    end
    

    subgraph "User Interface"
        S[Streamlit App]
    end
    
    K --> PG
    PG --> OS
    S --> OS
    S --> PG
    
    style PG fill:#c8e6c9
    style OS fill:#fff9c4
    style S fill:#e1bee7
```

## Sequence Diagram (User Query Flow)

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit
    participant OS as OpenSearch
    participant PG as PostgreSQL
    
    User->>UI: "Find dark thrillers"
    UI->>OS: Vector search
    OS-->>UI: Top 10 results
    UI->>PG: Get full details
    PG-->>UI: Movie data
    UI->>User: Display results
```

## Timeline

```mermaid
gantt
    title Project Timeline
    dateFormat YYYY-MM-DD
    section Week 1
    Setup Docker           :done, 2024-01-01, 2d
    Design Schema          :done, 2024-01-03, 2d
    section Week 2
    Build ETL              :active, 2024-01-05, 3d
    Test Pipeline          :2024-01-08, 2d
```

-