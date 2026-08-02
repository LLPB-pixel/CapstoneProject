#!/usr/bin/env python3
"""
Script para levantar el servidor Prompt Guard.
Uso: python start_server.py [--port 8000]
"""

import sys
import os

# Asegurar que src este en el path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Intentar importar desde src
try:
    from api.main import app
    print("Cargado desde src/api/main.py")
except ImportError as e:
    print(f"Error al cargar: {e}")
    print("\nIntentando metodo alternativo...")
    sys.exit(1)

import uvicorn

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Iniciar servidor Prompt Guard')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--no-reload', action='store_true')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  Prompt Guard - Sistema de Seguridad para LLMs")
    print("=" * 70)
    print(f"\n  Servidor: http://{args.host}:{args.port}")
    print("\n  Endpoints:")
    print("    http://localhost:8000/docs          - Swagger UI")
    print("    http://localhost:8000/detector      - Detector interactivo")
    print("    http://localhost:8000/chat          - Chat con IA")
    print("    http://localhost:8000/dashboard     - Dashboard")
    print("\n  Variables requeridas: MISTRAL_API_KEY, GROQ_API_KEY, JWT_SECRET_KEY")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=not args.no_reload)
