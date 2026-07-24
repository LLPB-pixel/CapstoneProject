#!/bin/bash

# Script para ejecutar el demo completo del sistema
# Uso: ./run_demo.sh [MISTRAL_API_KEY]

set -e

echo "=========================================="
echo "  Prompt Guard - Demo Mode"
echo "=========================================="
echo ""

if [ -z "$1" ]; then
    echo "No se proporciono API Key de Mistral"
    echo "Usando modo SIMULACION (sin backend real)"
    echo ""
    echo "Para usar el backend real:"
    echo "  ./run_demo.sh TU_API_KEY_MISTRAL"
    echo ""
    
    # Iniciar servidor simple para frontend
    echo "Iniciando servidor web en http://localhost:3000"
    echo "(Abre este enlace en tu navegador)"
    echo ""
    echo "  Landing:   http://localhost:3000/"
    echo "  Detector:  http://localhost:3000/detector"
    echo "  Chat IA:   http://localhost:3000/chat"
    echo "  Dashboard: http://localhost:3000/dashboard"
    echo "  Login:     http://localhost:3000/login"
    echo ""
    
    python3 serve_frontend.py 3000
else
    API_KEY=$1
    echo "API Key proporcionada"
    echo ""
    
    # Verificar si FastAPI esta instalado
    if python3 -c "import fastapi" 2>/dev/null; then
        echo "Iniciando backend con FastAPI..."
        cd 3-LLM-judge
        nohup python3 api_server.py "$API_KEY" --port 8000 > /tmp/backend.log 2>&1 &
        BACKEND_PID=$!
        sleep 3
        
        echo "Backend iniciado en http://localhost:8000 (PID: $BACKEND_PID)"
        echo ""
        echo "  Landing:   http://localhost:8000/"
        echo "  Detector:  http://localhost:8000/detector"
        echo "  Chat IA:   http://localhost:8000/chat"
        echo "  Dashboard: http://localhost:8000/dashboard"
        echo "  Login:     http://localhost:8000/login"
        echo "  API Docs:  http://localhost:8000/docs"
        echo ""
        
        # Esperar a que el usuario cierre el servidor
        echo "Presiona Ctrl+C para detener el servidor..."
        wait $BACKEND_PID
    else
        echo "FastAPI no esta instalado"
        echo "Instalalo con: pip install fastapi uvicorn"
        echo ""
        echo "Usando modo SIMULACION (solo frontend)"
        python3 -m http.server 3000 --bind 0.0.0.0
    fi
fi
