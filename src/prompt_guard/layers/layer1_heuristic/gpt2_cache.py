"""
Cache local para el modelo GPT-2.

Evita re-descargas innecesarias guardando el modelo en un directorio local.
Si el modelo ya está guardado, se carga directamente desde ahí.
Además mantiene el modelo cargado en memoria para reutilizarlo entre llamadas
(evita recargar el modelo en cada request o instanciación del filtro).
"""

import os
import logging

logger = logging.getLogger(__name__)

# Directorio de cache local relativo a este archivo
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2_local_cache")

# Cache en memoria: reutiliza el modelo entre llamadas (evita recargas costosas)
_LOADED_MODELS = {}


def get_gpt2(model_name: str = "gpt2", device: str = "cpu"):
    """
    Carga el modelo GPT-2 y tokenizer, reutilizando el cache local si existe.

    El modelo cargado se reutiliza en memoria entre llamadas, de modo que solo
    se carga/descarga una vez por proceso.

    Args:
        model_name: Nombre del modelo (default: "gpt2")
        device: Dispositivo torch ("cpu" o "cuda")

    Returns:
        Tupla (model, tokenizer)
    """
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as e:
        raise ImportError(
            "GPT-2 model requires transformers and torch. "
            "Install with: pip install torch transformers"
        ) from e

    local_path = os.path.join(_CACHE_DIR, model_name)

    if os.path.isdir(local_path) and os.listdir(local_path):
        logger.info(f"Cargando modelo GPT-2 desde cache local: {local_path}")
        tokenizer = GPT2TokenizerFast.from_pretrained(local_path)
        model = GPT2LMHeadModel.from_pretrained(local_path).to(device)
    else:
        logger.info(f"Modelo GPT-2 no encontrado en cache. Descargando '{model_name}'...")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

        os.makedirs(local_path, exist_ok=True)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        logger.info(f"Modelo GPT-2 guardado en cache local: {local_path}")

    model.eval()
    _LOADED_MODELS[model_name] = (model, tokenizer)
    return _LOADED_MODELS[model_name]
