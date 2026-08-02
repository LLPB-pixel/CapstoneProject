"""
Servidor API para el Pipeline de Deteccion de Prompt Injection
==============================================================

Este servidor expone el pipeline como una API REST.

Endpoints:
    GET  /                    - Pagina principal
    GET  /detector            - Detector de prompts
    GET  /chat                - Chat con IA protegida
    GET  /dashboard           - Dashboard de ataques
    POST /api/detect          - Analiza un prompt
    POST /api/chat            - Envia mensaje al chat
    POST /api/auth/register   - Registro de usuario
    POST /api/auth/login      - Login de usuario
    GET  /api/dashboard/*     - Estadisticas del dashboard
    GET  /health              - Health check
"""

import argparse
import time
import logging
import os
from pathlib import Path
from typing import Optional

# Configuracion
DEFAULT_MODEL_PATH = "./data/models/distilbert"
DEFAULT_PORT = 8000
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "templates"
FRONTEND_INDEX_PATH = FRONTEND_DIR / "index.html"
FRONTEND_LOGIN_PATH = FRONTEND_DIR / "login.html"
FRONTEND_DETECTOR_PATH = FRONTEND_DIR / "detector.html"
FRONTEND_DASHBOARD_PATH = FRONTEND_DIR / "dashboard.html"
FRONTEND_CHAT_PATH = FRONTEND_DIR / "chat.html"

# Variables globales
SERVER_API_KEY = None
SERVER_MODEL_PATH = None
SERVER_GROQ_KEY = None

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar pipeline y database desde la nueva estructura
try:
    from prompt_guard.pipeline import run_pipeline, MODEL_PATH as PIPELINE_MODEL_PATH
except ImportError as e:
    logger.error(f"No se pudo importar pipeline: {e}")
    # Crear una funcion dummy para no romper
    def run_pipeline(prompt, api_key, groq_key=None):
        return {
            'prompt': prompt,
            'final_verdict': 'CLEAN',
            'layer1': {'is_suspicious': False, 'risk_score': 0.1, 'triggered_categories': []},
            'layer2': {'label': 'benign', 'confidence': 0.9, 'score': 0.1},
            'layer3': {'is_good': True, 'score': 8.5, 'evaluation': 'Prompt seguro'},
            'detected_count': 0,
            'processing_time': 0.1
        }
    PIPELINE_MODEL_PATH = DEFAULT_MODEL_PATH

try:
    from prompt_guard.utils.database import (
        init_db, decode_token, register_user, create_token, 
        authenticate_user, log_attack, get_dashboard_stats,
        get_recent_attacks, get_attacks_timeline, get_top_source_ips,
        get_category_stats, get_layer_detection_stats, clear_attacks, get_db_path
    )
except ImportError as e:
    logger.error(f"No se pudo importar database: {e}")
    # Funciones dummy
    def init_db(): pass
    def decode_token(token): return None
    def register_user(email, password, name): return {"ok": False, "error": "DB not available"}
    def create_token(user): return "dummy_token"
    def authenticate_user(email, password): return None
    def log_attack(**kwargs): pass
    def get_dashboard_stats(user_email): return {}
    def get_recent_attacks(user_email, limit=50): return []
    def get_attacks_timeline(user_email, days=30): return []
    def get_top_source_ips(user_email, limit=10): return []
    def get_category_stats(user_email): return []
    def get_layer_detection_stats(user_email): return []
    def clear_attacks(user_email): pass
    def get_db_path(): return "./data/database/attacks.db"

# Configurar ruta del modelo
def set_model_path(path: str):
    global PIPELINE_MODEL_PATH
    PIPELINE_MODEL_PATH = path


def create_app(api_key: Optional[str] = None, model_path: Optional[str] = None,
               groq_key: Optional[str] = None):
    """Crea la aplicacion FastAPI."""
    try:
        from fastapi import FastAPI, HTTPException, Request, Query, Header
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI no esta instalado. Instalalo con: pip install fastapi uvicorn")
        raise

    # Obtener parametros
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
    init_db()
    
    app = FastAPI(
        title="Prompt Guard API",
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

    # Modelos
    class PromptRequest(BaseModel):
        prompt: str
        user_id: Optional[str] = None

    class RegisterRequest(BaseModel):
        email: str
        password: str
        name: str

    class LoginRequest(BaseModel):
        email: str
        password: str

    class ChatRequest(BaseModel):
        message: str
        history: Optional[list] = None

    # Funciones auxiliares
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

    def _get_user_from_token(authorization: Optional[str] = Header(None)) -> Optional[dict]:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        token = authorization.split(" ", 1)[1]
        return decode_token(token)

    # Rutas del frontend
    @app.get("/", response_class=HTMLResponse)
    async def serve_landing():
        return _serve_html(FRONTEND_INDEX_PATH)

    @app.get("/login", response_class=HTMLResponse)
    async def serve_login():
        return _serve_html(FRONTEND_LOGIN_PATH)

    @app.get("/detector", response_class=HTMLResponse)
    async def serve_detector():
        return _serve_html(FRONTEND_DETECTOR_PATH)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def serve_dashboard():
        return _serve_html(FRONTEND_DASHBOARD_PATH)

    @app.get("/chat", response_class=HTMLResponse)
    async def serve_chat():
        return _serve_html(FRONTEND_CHAT_PATH)

    # Rutas .html para compatibilidad
    @app.get("/{page}.html", response_class=HTMLResponse)
    async def serve_html_pages(page: str):
        pages = {
            "index": FRONTEND_INDEX_PATH,
            "login": FRONTEND_LOGIN_PATH,
            "detector": FRONTEND_DETECTOR_PATH,
            "dashboard": FRONTEND_DASHBOARD_PATH,
            "chat": FRONTEND_CHAT_PATH,
        }
        if page in pages:
            return _serve_html(pages[page])
        raise HTTPException(status_code=404, detail="Pagina no encontrada")

    # Autenticacion
    @app.post("/api/auth/register")
    async def register(req: RegisterRequest):
        result = register_user(req.email, req.password, req.name)
        if not result["ok"]:
            raise HTTPException(status_code=400, detail=result["error"])
        user = authenticate_user(req.email, req.password)
        token = create_token(user)
        return {"token": token, "user": {"email": user["email"], "name": user["name"]}}

    @app.post("/api/auth/login")
    async def login(req: LoginRequest):
        user = authenticate_user(req.email, req.password)
        if not user:
            raise HTTPException(status_code=401, detail="Email o contrasena incorrectos")
        token = create_token(user)
        return {"token": token, "user": {"email": user["email"], "name": user["name"]}}

    # Deteccion
    @app.post("/api/detect")
    async def detect_prompt(request: PromptRequest, req: Request,
                            authorization: Optional[str] = Header(None)):
        start_time = time.time()
        user_email = None
        user_data = _get_user_from_token(authorization)
        if user_data:
            user_email = user_data.get("email")

        try:
            # Ejecutar pipeline
            result = run_pipeline(request.prompt, api_key, groq_key=groq_key)
            processing_time = time.time() - start_time
            result["processing_time"] = round(processing_time, 4)

            # Registrar en DB
            try:
                log_attack(
                    result=result,
                    source_ip=_get_client_ip(req),
                    user_agent=_get_user_agent(req),
                    user_email=user_email,
                )
            except Exception as db_err:
                logger.warning(f"Error al registrar ataque en DB: {db_err}")

            logger.info(f"Prompt analizado - User: {user_email or 'unknown'}, Verdict: {result['final_verdict']}")
            return result

        except Exception as e:
            logger.error(f"Error procesando prompt: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    # Chat
    @app.post("/api/chat")
    async def chat_endpoint(request: ChatRequest, req: Request,
                           authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None

        try:
            # Analizar el mensaje con el pipeline
            result = run_pipeline(request.message, api_key, groq_key=groq_key)

            if result['final_verdict'] == 'BLOCKED':
                return {
                    "response": "Lo siento, pero he detectado un intento de prompt injection. Este mensaje ha sido bloqueado por razones de seguridad.",
                    "is_blocked": True,
                    "verdict": result
                }
            else:
                # Por ahora, respuesta simple
                return {
                    "response": f"Mensaje recibido: {request.message[:100]}. (Modo seguro: solo respuesta de prueba)",
                    "is_blocked": False,
                    "verdict": result
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    # Dashboard
    @app.get("/api/dashboard/stats")
    async def get_stats(authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None
        return get_dashboard_stats(user_email)

    @app.get("/api/dashboard/recent")
    async def get_recent(authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None
        return get_recent_attacks(user_email, limit=50)

    @app.get("/api/dashboard/timeline")
    async def get_timeline(authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None
        return get_attacks_timeline(user_email, days=30)

    @app.get("/api/dashboard/top-ips")
    async def get_top_ips(authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None
        return get_top_source_ips(user_email, limit=10)

    @app.get("/api/dashboard/categories")
    async def get_categories(authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None
        return get_category_stats(user_email)

    @app.get("/api/dashboard/layers")
    async def get_layers(authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None
        return get_layer_detection_stats(user_email)

    @app.delete("/api/dashboard/clear")
    async def clear_dashboard(authorization: Optional[str] = Header(None)):
        user_data = _get_user_from_token(authorization)
        user_email = user_data.get("email") if user_data else None
        if not user_email:
            raise HTTPException(status_code=401, detail="No autorizado")
        clear_attacks(user_email)
        return {"message": "Dashboard limpiado"}

    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "db_path": get_db_path(),
            "model_path": PIPELINE_MODEL_PATH
        }

    @app.get("/stats")
    async def stats():
        return {
            "model_path": PIPELINE_MODEL_PATH,
            "status": "running"
        }

    return app


# Funcion principal
def main():
    parser = argparse.ArgumentParser(description="Servidor API para Prompt Guard")
    parser.add_argument("--mistral_key", type=str, default=None, help="API Key de Mistral")
    parser.add_argument("--groq_key", type=str, default=None, help="API Key de Groq")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Puerto del servidor")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Ruta al modelo")
    args = parser.parse_args()

    # Configurar variables globales
    global SERVER_API_KEY, SERVER_MODEL_PATH, SERVER_GROQ_KEY
    SERVER_API_KEY = args.mistral_key or os.environ.get("MISTRAL_API_KEY")
    SERVER_GROQ_KEY = args.groq_key or os.environ.get("GROQ_API_KEY")
    SERVER_MODEL_PATH = args.model_path or os.environ.get("DISTILBERT_MODEL_PATH", DEFAULT_MODEL_PATH)

    # Crear y ejecutar la app
    app = create_app(api_key=SERVER_API_KEY, model_path=SERVER_MODEL_PATH, groq_key=SERVER_GROQ_KEY)

    print("=" * 70)
    print("Prompt Guard API - Servidor iniciado")
    print("=" * 70)
    print(f"Puerto: {args.port}")
    print(f"Modelo: {SERVER_MODEL_PATH}")
    print("=" * 70)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=True)


if __name__ == "__main__":
    main()

# Crear instancia de la app por defecto
app = create_app()
