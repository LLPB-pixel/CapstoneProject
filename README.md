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
│   ├── 📄 collect_datasets.py         — Descarga datasets de HuggingFace + locales
│   ├── 📄 convertAIprompts.py         — Convierte prompts de fuentes IA
│   ├── 📄 data_prep.py               — Genera splits train/val/test balanceados
│   ├── 📄 emergency_dataset_repair.py — Repara el dataset en caso de error
│   ├── 📄 filter_language.py          — Filtra por idioma y genera histogramas
│   └── 📄 merge_prompts.py            — Fusiona múltiples fuentes de prompts
│
├── 📁 1-heuristic_filters/      — 🔍 Capa 1: Filtros Heurísticos (~50ms)
│   ├── 📄 heuristic_filter.py        — Motor principal de detección por reglas
│   ├── 📄 perplexity_analysis.py     — Analiza perplexity con GPT-2
│   ├── 📄 calibrate_perplexity.py    — Calibra umbrales de perplexity
│   ├── 📄 calibrate_tfidf.py         — Calibra el modelo TF-IDF
│   ├── 📄 baseline_tfidf.py          — Baseline TF-IDF + Regresión Logística
│   ├── 📄 benchmark_layer1.py        — Benchmarks de rendimiento Capa 1
│   └── 📄 tfidf_model.joblib         — Modelo TF-IDF serializado
│
├── 📁 2-fase2/                  — 🧠 Capa 2: Modelos ML (~100-300ms)
│   ├── 📄 train_distilbert.py        — Fine-tuning de DistilBERT
│   ├── 📄 train_bert.py             — Fine-tuning de DeBERTa-v3 con MOF
│   ├── 📄 test_distilbert.py         — Tests del modelo DistilBERT
│   └── 📄 test_deberta.py            — Tests del modelo DeBERTa
│
├── 📁 3-LLM-judge/              — ⚖️ Capa 3: LLM-Judge + API (~1-3s)
│   ├── 📄 api_server.py              — Servidor FastAPI principal
│   ├── 📄 pipeline.py               — Pipeline de 3 capas completo
│   ├── 📄 LLM_evaluation.py          — Evaluación semántica vía Mistral/Groq
│   ├── 📄 distilbert_inference.py    — Inferencia del modelo DistilBERT
│   ├── 📄 database.py               — Base de datos SQLite + JWT auth
│   └── 📄 chat_api.py               — API de chat protegido con IA
│
├── 📁 frontend/                 — 🖥️ Interfaz de usuario
│   ├── 📄 index.html                — Landing page del proyecto
│   ├── 📄 detector.html             — Detector interactivo de prompts
│   ├── 📄 chat.html                 — Chat con IA protegido
│   ├── 📄 dashboard.html            — Dashboard de análisis de ataques
│   └── 📄 login.html                — Login y registro de usuarios
│
├── 📁 Data/                     — 📊 Datasets de entrenamiento
│   ├── 📄 combined_prompts_dataset.csv      — Dataset combinado completo
│   ├── 📄 def_combined_prompts_dataset.csv  — Dataset definitivo
│   ├── 📄 train.csv / val.csv / test.csv    — Splits para ML
│
├── 📁 models/                   — 🤖 Modelos entrenados guardados
│   └── deberta_detector/                  — Pesos del modelo DeBERTa
│
├── 📁 database/                 — 💾 Base de datos en ejecución
│   └── attacks.db                         — Registro de ataques detectados
│
├── 📁 docker/                   — 🐳 Despliegue contenedorizado
│   └── docker-compose.yml                 — Orquestación de servicios
│
├── 📁 advbench/                 — 📋 Benchmark de ataques dañinos
│   ├── 📄 harmful_behaviors.csv           — Comportamientos dañinos
│   └── 📄 harmful_strings.csv             — Strings maliciosos conocidos
│
└── 📁 jaqilbreakbench/          — 📋 Benchmark de jailbreak
    ├── 📄 harmful-behaviors.csv           — Prompts maliciosos
    ├── 📄 benign-behaviors.csv            — Prompts seguros
    └── 📄 judge-comparison.csv            — Comparación de modelos juez
```

---

## ⚙️ Arquitectura del Sistema (3 Capas)

```
  Usuario → ┌────────────────────────────┐
             │  CAPA 1: Heurístico       │  ← Regex, Base64, Perplexity
             └──────────────┬─────────────┘  (~50ms, CPU-only)
                            ↓
             ┌────────────────────────────┐
             │  CAPA 2: DistilBERT/DeBERTa│  ← Modelo ML fine-tuneado
             └──────────────┬─────────────┘  (~200ms, GPU opcional)
                            ↓
             ┌────────────────────────────┐
             │  CAPA 3: LLM-Judge         │  ← Mistral/Groq como juez
             └──────────────┬─────────────┘  (~2s, API externa)
                            ↓
             ┌────────────────────────────┐
             │  VOTACIÓN POR MAYORÍA (2/3) │  → 🟢 LIMPIO o 🔴 BLOQUEADO
             └────────────────────────────┘
```

**Filosofía**: Defensa en profundidad — si una capa falla, las demás compensan. En caso de empate → **fail-safe** (bloquea por precaución).

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML** | PyTorch, HuggingFace Transformers |
| **Modelos** | DistilBERT (rápido), DeBERTa-v3 (preciso) |
| **APIs** | Mistral (principal), Groq (fallback) |
| **DB** | SQLite + JWT (python-jose) |
| **Frontend** | HTML5, JS vanilla, Chart.js |
| **Deploy** | Docker, Docker Compose |
