#!/bin/bash
# Script para levantar el servidor Prompt Guard

# Activar entorno virtual curso
source /home/llorenc/anaconda3/bin/activate /home/llorenc/anaconda3/envs/curso

# Navegar al directorio del proyecto
cd /home/llorenc/Desktop/SamsungIA/CapstoneProject/src

# Exportar variables de entorno si no están configuradas
export MISTRAL_API_KEY="${MISTRAL_API_KEY:-}"
export GROQ_API_KEY="${GROQ_API_KEY:-}"
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-$(python -c 'import secrets; print(secrets.token_urlsafe(64))')}"

echo "========================================="
echo "  Prompt Guard - Servidor"
echo "========================================="
echo "Puerto: 8000"
echo "Presiona Ctrl+C para detener"
echo "========================================="
echo ""

# Iniciar servidor
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
