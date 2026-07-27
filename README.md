# 🛡️ Prompt Guard — Sistema de Seguridad para LLMs

**Capstone Project Samsung** — Detecta y bloquea ataques de prompt injection en modelos de lenguaje usando 3 capas de defensa en profundidad.

---

## 📁 Estructura del Repositorio

```
CapstoneProject/
│
├── 📄 .env.example              — Variables de entorno necesarias
├── 📄 Dockerfile                — Imagen Docker del sistema
├── 📄 requirements.txt          — Todas las dependencias del proyecto
├── 📄 requirements-prod.txt     — Dependencias mínimas para producción
├── 📄 requirements-dev.txt      — Dependencias de desarrollo
├── 📄 serve_frontend.py         — Servidor estático para el frontend
├── 📄 train_distilbert.py       — Wrapper para entrenar DistilBERT
├── 📄 data_sources.md           — Fuentes de datos académicas
│
├── 📁 0-data_collection_generation/  — 🔄 Recolección y preparación de datos
│   ├── 📄 collect_datasets.py         — Descarga datasets de HuggingFace + locales y los unifica
│   ├── 📄 convertAIprompts.py         — Convierte CSVs de prompts IA al formato estándar
│   ├── 📄 data_prep.py               — Genera splits train/val/test balanceados 50/50
│   ├── 📄 emergency_dataset_repair.py — Repara labels inválidos y mapea a 0/1
│   ├── 📄 filter_language.py          — Filtra por idioma, elimina emojis y genera histogramas
│   ├── 📄 merge_prompts.py            — Fusiona dos CSVs eliminando duplicados
│   └── 📄 prompts_histogram.png       — Histograma de distribución de prompts benignos vs maliciosos
│
├── 📁 1-heuristic_filters/      — 🔍 Capa 1: Filtros Heurísticos (~0-50ms)
│   ├── 📄 heuristic_filter.py        — Motor consolidado: regex + base64 + zero-width + homoglifos + perplexity + TF-IDF
│   ├── 📄 perplexity_analysis.py     — Calcula perplexity con GPT-2 y encuentra cutoff óptimo por entropía
│   ├── 📄 calibrate_perplexity.py    — Calibra umbral de perplexity (percentil 97.5 de prompts buenos)
│   ├── 📄 calibrate_tfidf.py         — Entrena y serializa modelo TF-IDF + LogisticRegression
│   ├── 📄 baseline_tfidf.py          — Baseline TF-IDF con métricas y top-20 n-gramas predictivos
│   ├── 📄 benchmark_layer1.py        — Benchmark completo: accuracy, latencia p50/p95/p99, matriz de confusión
│   ├── 📄 example_perplexity_analysis.py — Ejemplo práctico de análisis de perplexity con datos sintéticos
│   ├── 📄 gpt2_model_cache.py        — Cache local del modelo GPT-2 para evitar re-descargas
│   ├── 📄 test_heuristic_filter.py   — Tests unitarios comprehensivos del filtro heurístico
│   ├── 📄 test_examples.py           — Tests de los ejemplos del frontend contra el filtro
│   ├── 📄 test_benchmark_tfidf_load.py — Benchmark de tiempo de carga del modelo TF-IDF
│   ├── 📄 tfidf_model.joblib         — Modelo TF-IDF serializado con joblib
│   ├── 📄 tfidf_stats.json           — Métricas del TF-IDF (accuracy, F1, ROC-AUC)
│   ├── 📄 perplexity_threshold.json  — Umbral calibrado de perplexity + estadísticas
│   ├── 📄 mock_entropy_curve.png     — Curva de entropía con datos simulados
│   └── 📄 mock_histogram.png         — Histograma de perplexity con datos simulados
│
├── 📁 2-fase2/                  — 🧠 Capa 2: Modelos ML (~100-300ms)
│   ├── 📄 train_distilbert.py        — Fine-tuning de DistilBERT (rápido, ligero)
│   ├── 📄 train_bert.py             — Fine-tuning de DeBERTa-v3-base con técnica MOF
│   ├── 📄 test_distilbert.py         — Tests del modelo DistilBERT
│   ├── 📄 test_deberta.py            — Tests del modelo DeBERTa
│   ├── 📄 test_train_distilbert.py   — Tests del proceso de entrenamiento
│   ├── 📄 requirements.txt           — Dependencias específicas de la fase 2
│   └── 📄 logs.txt / newlogs*.txt    — Logs de entrenamiento
│
├── 📁 3-LLM-judge/              — ⚖️ Capa 3: LLM-Judge + API (~1-3s)
│   ├── 📄 api_server.py              — Servidor FastAPI principal con endpoints de detección y auth
│   ├── 📄 pipeline.py               — Pipeline de 3 capas con votación por mayoría
│   ├── 📄 LLM_evaluation.py          — Evaluación semántica vía Mistral API (fallback: Groq)
│   ├── 📄 distilbert_inference.py    — Inferencia del modelo DistilBERT para Capa 2
│   ├── 📄 database.py               — Base de datos SQLite + autenticación JWT
│   └── 📄 chat_api.py               — API de chat con IA protegida por las 3 capas
│
├── 📁 frontend/                 — 🖥️ Interfaz de usuario
│   ├── 📄 index.html                — Landing page del proyecto
│   ├── 📄 detector.html             — Detector interactivo de prompts con ejemplos predefinidos
│   ├── 📄 chat.html                 — Chat con IA protegido por las 3 capas
│   ├── 📄 dashboard.html            — Dashboard de análisis de ataques con Chart.js
│   └── 📄 login.html                — Login y registro de usuarios
│
├── 📁 Data/                     — 📊 Datasets de entrenamiento
│   ├── 📄 combined_prompts_dataset.csv      — Dataset combinado de todas las fuentes
│   ├── 📄 filtered_prompts_dataset.csv      — Dataset filtrado por idioma y sin emojis
│   ├── 📄 def_combined_prompts_dataset.csv  — Dataset definitivo deduplicado
│   ├── 📄 hf_datasets_prompt_export.csv     — Export de HuggingFace datasets
│   └── 📄 train.csv / val.csv / test.csv    — Splits finales para ML (stratificados 70/15/15)
│
├── 📁 models/                   — 🤖 Modelos entrenados guardados
│   └── deberta_detector/                  — Pesos del modelo DeBERTa
│
├── 📁 database/                 — 💾 Base de datos en ejecución
│   └── attacks.db                         — Registro de ataques detectados
│
├── 📁 docker/                   — 🐳 Despliegue contenedorizado
│   ├── 📄 docker-compose.yml              — Orquestación de servicios
│   └── 📄 CAPROVER_DEPLOYMENT.md          — Guía de despliegue en CapRover
│
├── 📁 advbench/                 — 📋 Benchmark de ataques dañinos
│   ├── 📄 harmful_behaviors.csv           — Comportamientos dañinos (AdvBench)
│   └── 📄 harmful_strings.csv             — Strings maliciosos conocidos
│
└── 📁 jaqilbreakbench/          — 📋 Benchmark de jailbreak
    ├── 📄 harmful-behaviors.csv           — Prompts maliciosos (JailbreakBench)
    ├── 📄 benign-behaviors.csv            — Prompts seguros
    └── 📄 judge-comparison.csv            — Comparación de modelos juez
```

---

## 🔄 Pipeline de Datos: Qué se hizo

El dataset fue construido mediante un proceso de **4 fases**:

### 1. Recolección (`collect_datasets.py`)
Se descargaron **3 datasets de HuggingFace** (AdvBench, Prompt-Injections, JailbreakHub) y se combinaron con **10+ archivos locales** (red teaming, AdvBench local, JailbreakBench). Todos se unificaron en un CSV con formato `prompt;label;source`.

### 2. Limpieza y Filtrado (`filter_language.py`, `merge_prompts.py`)
- Se **eliminaron emojis** de todos los prompts
- Se **filtró por idioma** (inglés, español, francés, alemán) detectando palabras clave
- Se **fusionaron** CSVs temporales eliminando duplicados exactos
- Se generaron **histogramas** de distribución de longitud por clase

### 3. Balanceo y Split (`data_prep.py`, `emergency_dataset_repair.py`)
- Se **balancearon las clases** 50/50 (benignos de Alpaca + maliciosos del dataset combinado)
- Se crearon **splits estratificados**: 70% train / 15% val / 15% test
- Se **repararon labels inválidos** (unknown → eliminados, good/bad → 0/1)

### 4. Calibración (`calibrate_*.py`)
- Se calibró el **umbral de perplexity** en el percentil 97.5 de prompts buenos
- Se entrenó y serializó el **modelo TF-IDF** con métricas (accuracy, F1, ROC-AUC)

**Resultado final**: Dataset balanceado con prompts benignos (instrucciones genéricas) y maliciosos (jailbreak, prompt injection, red teaming) listo para entrenar los modelos de las 3 capas.

---

## ⚙️ Arquitectura del Sistema (3 Capas)

```
  Usuario → ┌────────────────────────────┐
             │  CAPA 1: Heurístico       │  ← Regex, Base64, Perplexity, TF-IDF
             └──────────────┬─────────────┘  (~0-50ms, CPU-only)
                            ↓
             ┌────────────────────────────┐
             │  CAPA 2: DistilBERT/DeBERTa│  ← Modelo ML fine-tuneado
             └──────────────┬─────────────┘  (~100-300ms, GPU opcional)
                            ↓
             ┌────────────────────────────┐
             │  CAPA 3: LLM-Judge         │  ← Mistral/Groq como juez semántico
             └──────────────┬─────────────┘  (~1-3s, API externa)
                            ↓
             ┌────────────────────────────┐
             │  VOTACIÓN POR MAYORÍA (2/3) │  → 🟢 LIMPIO o 🔴 BLOQUEADO
             └────────────────────────────┘
```

**Filosofía**: Defensa en profundidad — si una capa falla, las demás compensan. En caso de empate → **fail-safe** (bloquea por precaución).

### Capa 1 — Filtro Heurístico (heuristic_filter.py)
- **Regex/keywords**: ~50 patrones de jailbreak en 4 idiomas (inglés, español, francés, alemán)
- **Encoding tricks**: Detección de Base64, caracteres zero-width, homoglifos unicode
- **Perplexity**: scoring con GPT-2 para detectar texto generado automáticamente
- **TF-IDF**: baseline con Regresión Logística como referencia

### Capa 2 — Modelo ML
- **DistilBERT**: más rápido,适合 para latencia baja
- **DeBERTa-v3-base**: más preciso, con técnica MOF para reducir falsos positivos

### Capa 3 — LLM-Judge
- **Mistral API**: análisis semántico profundo del prompt
- **Groq API**: fallback automático si Mistral no está disponible

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML** | PyTorch, HuggingFace Transformers |
| **Modelos** | DistilBERT (rápido), DeBERTa-v3 (preciso), GPT-2 (perplexity) |
| **APIs** | Mistral (principal), Groq (fallback) |
| **DB** | SQLite + JWT (python-jose) |
| **Frontend** | HTML5, JS vanilla, Chart.js, Font Awesome 6 |
| **Deploy** | Docker, Docker Compose, CapRover |
| **Experiment Tracking** | Weights & Biases (wandb) |
