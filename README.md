# Prompt Guard - Docker Deployment

## Quick Start

### Build and Run

```bash
# Build the Docker image
docker build -t prompt-guard .

# Run with demo mode (no API key needed)
docker run -d \
  -p 8000:8000 \
  --name prompt-guard \
  prompt-guard

# Or with Mistral API key for full functionality
docker run -d \
  -p 8000:8000 \
  -e MISTRAL_API_KEY="your_key_here" \
  -v /path/to/models:/app/models \
  --name prompt-guard \
  prompt-guard
```

### Using Docker Compose

```bash
# Create .env file
cp .env.example .env
nano .env  # Add your MISTRAL_API_KEY

# Build and run
docker-compose build
docker-compose up -d
```

## Access

- **Backend API**: `http://your-server:8000`
- **API Docs**: `http://your-server:8000/docs`
- **Health Check**: `http://your-server:8000/health`

## Configuration

### Environment Variables

Create `.env` file:
```env
# Mistral API Key (required for full functionality)
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Volumes

Mount your models directory:
```yaml
volumes:
  - ./models:/app/models
```

## CapRover Deployment

1. **Build the image**:
   ```bash
   docker build -t your-registry/prompt-guard .
   ```

2. **Push to your registry**:
   ```bash
   docker push your-registry/prompt-guard
   ```

3. **Deploy on CapRover**:
   - Create new app
   - Select custom image
   - Enter: `your-registry/prompt-guard`
   - Set environment variable: `MISTRAL_API_KEY=your_key_here`
   - Add persistent storage for `/app/models`
   - Deploy

4. **Upload models**:
   - Use SFTP to upload models to `/var/lib/docker/volumes/caprover_prompt-guard/_data/models/`
   - Or use CapRover's file manager

## Frontend Deployment

The frontend is static HTML/JS and can be deployed separately:

1. **Upload frontend files** to your web server or CDN
2. **Configure API URL** in `frontend/index.html`:
   ```javascript
   const API_URL = 'https://your-caprover-app.your-domain.com/detect'
   ```
3. **Deploy frontend** to any static hosting (Netlify, Vercel, S3, etc.)

## Monitoring

```bash
# View logs
docker logs prompt-guard

# Check health
docker exec prompt-guard curl http://localhost:8000/health

# View resource usage
docker stats prompt-guard
```

## Notes

- Demo mode works without API key (simulated responses)
- Full mode requires Mistral API key
- Models should be mounted as volumes for persistence
- Backend runs on port 8000
- Frontend can be deployed separately as static files
