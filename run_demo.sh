#!/bin/bash

# Script para ejecutar el demo completo del sistema
# Uso: ./run_demo.sh [MISTRAL_API_KEY]

set -e

echo "=========================================="
echo "  Prompt Injection Detector - Demo Mode"
echo "=========================================="
echo ""

if [ -z "$1" ]; then
    echo "⚠️  No se proporcionó API Key de Mistral"
    echo "    Usando modo SIMULACIÓN (sin backend real)"
    echo ""
    echo "Para usar el backend real:"
    echo "  ./run_demo.sh TU_API_KEY_MISTRAL"
    echo ""
    
    # Iniciar servidor simple para frontend
    echo "Iniciando servidor web en http://localhost:3000"
    echo "(Abre este enlace en tu navegador)"
    echo ""
    
    cd frontend
    python3 -m http.server 3000 --bind 0.0.0.0
else
    API_KEY=$1
    echo "✅ API Key proporcionada"
    echo ""
    
    # Verificar si FastAPI está instalado
    if python3 -c "import fastapi" 2>/dev/null; then
        echo "Iniciando backend con FastAPI..."
        cd 3-LLM-judge
        nohup python3 api_server.py "$API_KEY" --port 8000 > /tmp/backend.log 2>&1 &
        BACKEND_PID=$!
        sleep 3
        
        echo "Backend iniciado en http://localhost:8000 (PID: $BACKEND_PID)"
        echo ""
        
        echo "Iniciando frontend en http://localhost:3000"
        cd ../frontend
        python3 -m http.server 3000 --bind 0.0.0.0
        
        # Matar backend al salir
        kill $BACKEND_PID 2>/dev/null || true
    else
        echo "⚠️  FastAPI no está instalado"
        echo "    Instálalo con: pip install fastapi uvicorn"
        echo ""
        echo "Usando modo SIMULACIÓN (solo frontend)"
        python3 -m http.server 3000 --bind 0.0.0.0
    fi
fi
