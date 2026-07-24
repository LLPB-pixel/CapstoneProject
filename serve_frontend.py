#!/usr/bin/env python3
"""
Servidor HTTP simple que sirve los archivos del frontend
mapeando URLs limpias (ej: /chat) a archivos .html (ej: /chat.html).
"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

CLEAN_URL_MAP = {
    "/chat": "/chat.html",
    "/detector": "/detector.html",
    "/dashboard": "/dashboard.html",
    "/login": "/login.html",
}


class CleanURLHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        if self.path in CLEAN_URL_MAP:
            self.path = CLEAN_URL_MAP[self.path]
        return super().do_GET()


if __name__ == "__main__":
    os.chdir(FRONTEND_DIR)
    server = http.server.HTTPServer(("0.0.0.0", PORT), CleanURLHandler)
    print(f"Sirviendo frontend en http://localhost:{PORT}")
    print(f"  Landing:   http://localhost:{PORT}/")
    print(f"  Detector:  http://localhost:{PORT}/detector")
    print(f"  Chat IA:   http://localhost:{PORT}/chat")
    print(f"  Dashboard: http://localhost:{PORT}/dashboard")
    print(f"  Login:     http://localhost:{PORT}/login")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
