"""
Servidor API para el Pipeline de Deteccion de Prompt Injection
==============================================================

Este servidor expone el pipeline como una API REST para ser consumida
por el frontend o cualquier otro cliente.

Uso:
    python api_server.py --mistral_key <MISTRAL_API_KEY> [--groq_key GROQ_API_KEY] [--port PUERTO] [--model_path RUTA_AL_MODELO]
    
Ejemplo:
    python api_server.py --mistral_key sk-1234567890 --groq_key gsk_xxx --port 8000 --model_path ../models/distilbert_sentinel

Endpoints:
    GET  /              - Pagina principal (landing page)
    GET  /detector      - Detector de prompts
    GET  /dashboard     - Dashboard de ataques
    POST /detect        - Analiza un prompt
    GET  /api/dashboard/stats       - Estadisticas generales
    GET  /api/dashboard/recent      - Ataques recientes
    GET  /api/dashboard/timeline    - Timeline de ataques
    GET  /api/dashboard/top-ips     - Top IPs atacantes
    GET  /api/dashboard/categories  - Categorias de ataque
    GET  /api/dashboard/layers      - Stats por capa
    GET  /health        - Health check
    GET  /stats         - Info del modelo
"""

import argparse
import time
import logging
import os
from pathlib import Path
from typing import Optional

# Configuracion
DEFAULT_MODEL_PATH = "./models/distilbert_sentinel/checkpoint-22797"
DEFAULT_PORT = 8000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_INDEX_PATH = FRONTEND_DIR / "index.html"
FRONTEND_DETECTOR_PATH = FRONTEND_DIR / "detector.html"
FRONTEND_DASHBOARD_PATH = FRONTEND_DIR / "dashboard.html"

# Variables globales para el servidor (usadas con reload)
SERVER_API_KEY = None
SERVER_MODEL_PATH = None
SERVER_GROQ_KEY = None

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar el pipeline
from pipeline import run_pipeline, MODEL_PATH as PIPELINE_MODEL_PATH


# Configurar ruta del modelo
def set_model_path(path: str):
    """Configura la ruta del modelo DistilBERT."""
    global PIPELINE_MODEL_PATH
    PIPELINE_MODEL_PATH = path
    import pipeline as pipe_module
    pipe_module.MODEL_PATH = path


def create_app(api_key: Optional[str] = None, model_path: Optional[str] = None,
               groq_key: Optional[str] = None):
    """Crea la aplicacion FastAPI."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI no esta instalado. Instalalo con: pip install fastapi uvicorn")
        raise

    # Obtener parametros de variables globales si no se proporcionan
    global SERVER_API_KEY, SERVER_MODEL_PATH, SERVER_GROQ_KEY
    if api_key is None:
        api_key = SERVER_API_KEY
    if model_path is None:
        model_path = SERVER_MODEL_PATH
    if groq_key is None:
        groq_key = SERVER_GROQ_KEY
    
    # Configurar ruta del modelo
    set_model_path(model_path)

    # Inicializar base de datos
    from database import init_db
    init_db()
    
    app = FastAPI(
        title="Prompt Injection Detection API",
        description="API para detectar prompt injection en modelos de lenguaje",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Modelos ---

    class PromptRequest(BaseModel):
        prompt: str
        user_id: Optional[str] = None

    # --- Funciones auxiliares ---

    def _serve_html(file_path: Path):
        if not file_path.exists():
            logger.error(f"Archivo no encontrado: {file_path}")
            raise HTTPException(status_code=404, detail="Pagina no encontrada")
        return FileResponse(file_path, media_type="text/html")

    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _get_user_agent(request: Request) -> str:
        return request.headers.get("user-agent", "unknown")

    # --- Rutas del frontend ---

    @app.get("/", response_class=HTMLResponse)
    async def serve_landing():
        """Pagina principal con explicacion del proyecto."""
        return _serve_html(FRONTEND_INDEX_PATH)

    @app.get("/detector", response_class=HTMLResponse)
    async def serve_detector():
        """Pagina del detector de prompts."""
        return _serve_html(FRONTEND_DETECTOR_PATH)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def serve_dashboard():
        """Dashboard de analisis de ataques."""
        return _serve_html(FRONTEND_DASHBOARD_PATH)

    # --- Endpoint de deteccion ---

    @app.post("/detect", response_model=dict)
    async def detect_prompt(request: PromptRequest, req: Request):
        """Analiza un prompt para detectar si contiene inyeccion de instrucciones."""
        start_time = time.time()

        try:
            print(f"\n{'='*60}")
            print(f"[API] Nuevo request - Prompt: \"{request.prompt[:100]}{'...' if len(request.prompt) > 100 else ''}\"")
            print(f"{'='*60}")

            # Ejecutar pipeline
            result = run_pipeline(request.prompt, api_key, groq_key=groq_key)
            
            processing_time = time.time() - start_time
            result["processing_time"] = round(processing_time, 4)

            # Registrar en la base de datos
            try:
                from database import log_attack
                log_attack(
                    result=result,
                    source_ip=_get_client_ip(req),
                    user_agent=_get_user_agent(req),
                )
            except Exception as db_err:
                logger.warning(f"Error al registrar ataque en DB: {db_err}")
            
            logger.info(
                f"Prompt analizado - User: {request.user_id or 'unknown'}, "
                f"Verdict: {result['final_verdict']}, "
                f"Time: {processing_time:.2f}s"
            )

            print(f"[API] Veredicto: {result['final_verdict']}", end="")
            if result['blocked_at_layer']:
                print(f" (bloqueado en capa {result['blocked_at_layer']})")
            else:
                print(" (limpio)")
            print(f"[API] Tiempo total: {processing_time:.2f}s")
            print(f"{'='*60}\n")

            return result
            
        except Exception as e:
            logger.error(f"Error procesando prompt: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error al procesar el prompt: {str(e)}"
            )

    # --- Endpoints del dashboard ---

    @app.get("/api/dashboard/stats")
    async def dashboard_stats():
        """Estadisticas generales del sistema."""
        from database import get_dashboard_stats
        return get_dashboard_stats()

    @app.get("/api/dashboard/recent")
    async def dashboard_recent(
        limit: int = Query(default=50, ge=1, le=500),
        verdict: Optional[str] = Query(default=None)
    ):
        """Ataques recientes registrados."""
        from database import get_recent_attacks
        return get_recent_attacks(limit=limit, verdict=verdict)

    @app.get("/api/dashboard/timeline")
    async def dashboard_timeline(
        days: int = Query(default=7, ge=1, le=90)
    ):
        """Timeline de ataques por dia."""
        from database import get_attacks_timeline
        return get_attacks_timeline(days=days)

    @app.get("/api/dashboard/top-ips")
    async def dashboard_top_ips(
        limit: int = Query(default=10, ge=1, le=50)
    ):
        """Principales IPs atacantes."""
        from database import get_top_source_ips
        return get_top_source_ips(limit=limit)

    @app.get("/api/dashboard/categories")
    async def dashboard_categories():
        """Estadisticas de categorias de ataque."""
        from database import get_category_stats
        return get_category_stats()

    @app.get("/api/dashboard/layers")
    async def dashboard_layers():
        """Estadisticas de deteccion por capa."""
        from database import get_layer_detection_stats
        return get_layer_detection_stats()

    @app.delete("/api/dashboard/clear")
    async def clear_dashboard():
        """Borra todos los registros del dashboard."""
        from database import clear_attacks
        clear_attacks()
        return {"status": "ok", "message": "Dashboard limpiado"}

    # --- Endpoints de sistema ---

    @app.get("/health")
    async def health_check():
        """Endpoint para monitorizacion."""
        from database import get_db_path
        db_exists = os.path.exists(get_db_path())
        return {
            "status": "ok",
            "version": "1.0.0",
            "model_path": PIPELINE_MODEL_PATH,
            "database": "connected" if db_exists else "not_initialized",
        }

    @app.get("/stats")
    async def get_stats():
        """Obtener informacion del modelo cargado."""
        from distilbert_inference import get_distilbert_classifier
        try:
            classifier = get_distilbert_classifier()
            return {
                "model_loaded": True,
                "model_path": str(classifier.model_path),
                "device": str(classifier.device)
            }
        except Exception as e:
            return {
                "model_loaded": False,
                "error": str(e)
            }

    return app


def run_server(api_key: str, port: int = DEFAULT_PORT, model_path: str = DEFAULT_MODEL_PATH,
               groq_key: Optional[str] = None):
    """Ejecuta el servidor API."""
    try:
        import uvicorn
    except ImportError:
        logger.error("Uvicorn no esta instalado. Instalalo con: pip install uvicorn")
        return

    global SERVER_API_KEY, SERVER_MODEL_PATH, SERVER_GROQ_KEY
    SERVER_API_KEY = api_key
    SERVER_MODEL_PATH = model_path
    SERVER_GROQ_KEY = groq_key
    
    logger.info(f"Iniciando servidor API en http://localhost:{port}")
    logger.info(f"Modelo DistilBERT: {model_path}")
    if SERVER_GROQ_KEY:
        logger.info("Groq API: Configurada (fallback para Mistral)")
    else:
        logger.warning("Groq API: No configurada (sin fallback si Mistral falla)")
    logger.info("Landing page: http://localhost:{}/".format(port))
    logger.info("Detector:      http://localhost:{}/detector".format(port))
    logger.info("Dashboard:     http://localhost:{}/dashboard".format(port))
    logger.info("API Docs:      http://localhost:{}/docs".format(port))
    
    uvicorn.run(
        "api_server:create_app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True,
        factory=True
    )


def main():
    """Funcion principal."""
    parser = argparse.ArgumentParser(
        description="Servidor API para deteccion de Prompt Injection"
    )
    parser.add_argument(
        "--mistral_key",
        type=str,
        required=True,
        help="API Key de Mistral para la Capa 3 (LLM-Judge)"
    )
    parser.add_argument(
        "--groq_key",
        type=str,
        default=None,
        help="API Key de Groq (fallback para Mistral)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Puerto del servidor (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta al modelo DistilBERT (default: {DEFAULT_MODEL_PATH})"
    )
    
    args = parser.parse_args()

    # Iniciar servidor
    run_server(args.mistral_key, args.port, args.model_path, groq_key=args.groq_key)


if __name__ == "__main__":
    main()
