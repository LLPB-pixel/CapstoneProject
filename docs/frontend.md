# Frontend - Prompt Injection Detector

Frontend sencilla en HTML/CSS/JS para visualizar el pipeline de deteccion de prompt injection.

## Estructura

```
frontend/
├── index.html    # Página principal con el frontend
└── README.md     # Este archivo
```

## Características

- **Diseño moderno y responsivo** con degradados y animaciones
- **Visualización en tiempo real** del progreso por capas
- **Ejemplos predefinidos** para probar rápidamente
- **Modo demo** integrado (funciona sin backend)
- **Notificaciones toast** para feedback al usuario
- **Estilo "chulo"** con:
  - Tarjetas para cada capa
  - Barras de progreso animadas
  - Colores semáforo (rojo/amarillo/verde)
  - Iconos de Font Awesome
  - Diseño adaptable a móvil

## Uso

### Opción 1: Modo Demo (sin backend)
Simply abre `index.html` en tu navegador. Funciona completamente en el cliente con datos simulados.

```bash
# En Linux/Mac
xdg-open index.html  # o simplemente haz doble click

# En Windows
start index.html
```

### Opción 2: Con Backend Real

1. **Iniciar el servidor API:**
   ```bash
   cd ../3-LLM-judge
   pip install fastapi uvicorn
   python api_server.py TU_API_KEY_MISTRAL --port 8000
   ```

2. **Configurar el frontend:**
   - Edita `index.html` y cambia la línea:
     ```javascript
     const API_URL = 'http://localhost:8000/detect';
     ```

3. **Abrir el frontend:**
   - Puedes usar un servidor simple:
     ```bash
     cd frontend
     python -m http.server 3000
     ```
   - Luego abre `http://localhost:3000` en tu navegador

### Opción 3: Despliegue Completo

```bash
# 1. Crear estructura
mkdir -p frontend
cp index.html frontend/

# 2. Instalar dependencias del backend
cd 3-LLM-judge
pip install fastapi uvicorn python-multipart

# 3. Iniciar backend (en un terminal)
python api_server.py TU_API_KEY --port 8000

# 4. Iniciar frontend (en otro terminal)
cd frontend
python -m http.server 3000 --bind 0.0.0.0
```

## Personalización

### Cambiar los ejemplos
Edita las líneas en `index.html`:
```javascript
example-chips.innerHTML = `
    <div class="example-chip" data-prompt="Nuevo ejemplo 1">Etiqueta 1</div>
    <div class="example-chip" data-prompt="Nuevo ejemplo 2">Etiqueta 2</div>
`;
```

### Cambiar los colores
Modifica las variables CSS al inicio del `<style>`:
```css
:root {
    --primary: #2563eb;      /* Azul */
    --success: #10b981;      /* Verde */
    --danger: #ef4444;       /* Rojo */
    --warning: #f59e0b;      /* Amarillo */
}
```

### Cambiar el logo/título
Modifica el `<header>` en `index.html`:
```html
<header>
    <h1><i class="fas fa-shield-halved"></i> Mi Detector</h1>
    <p>Mi subtítulo personalizado</p>
</header>
```

## Tecnologías Usadas

- **HTML5** - Estructura
- **CSS3** - Estilos con variables, flexbox, grid, animaciones
- **JavaScript (ES6+)** - Lógica del frontend
- **Font Awesome 6** - Iconos (cargado desde CDN)
- **FastAPI** (opcional) - Backend API
- **Uvicorn** (opcional) - Servidor ASGI

## Notas

1. **El modo demo** simula los resultados con valores aleatorios basados en el contenido del prompt. Es útil para desarrollo y demostraciones.

2. **Para producción**, se recomienda usar el backend real con el modelo entrenado.

3. **El frontend es completamente estático** y puede ser servido desde cualquier hosting (GitHub Pages, Netlify, Vercel, etc.).

4. **Si usas solo el frontend**, los datos son generados localmente y no se envían a ningún servidor.

## Ejemplo de Despliegue en Vercel

1. Crea un archivo `vercel.json`:
```json
{
  "version": 2,
  "builds": [
    {"src": "frontend/index.html", "use": "@vercel/static"}
  ],
  "routes": [
    {"src": "/(.*)", "dest": "frontend/index.html"}
  ]
}
```

2. Despliega:
```bash
vercel
```

## Solución de Problemas

### Los iconos no se ven
- Asegúrate de tener conexión a internet (Font Awesome se carga desde CDN)
- Alternativa: Descarga Font Awesome localmente

### El backend no responde
- Verifica que el servidor esté corriendo: `curl http://localhost:8000/health`
- Revisa que el puerto sea el correcto
- Comprueba que no haya firewall bloqueando el puerto

### CORS errors
- El backend ya tiene CORS configurado para todas las origines
- Si usas otro dominio, actualiza `allow_origins` en `api_server.py`
