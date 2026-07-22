"""
Script para eliminar un usuario y sus datos asociados de la base de datos.
=========================================================================

Elimina el usuario de la tabla 'users' y todos los registros de la tabla
'attacks' que tengan su email en user_email.

Uso:
    python delete_user.py                         # Elimina perezllorenc@gmail.com
    python delete_user.py otro@email.com          # Elimina otro@email.com
    python delete_user.py --list                  # Lista todos los usuarios
"""

import sqlite3
import os
import sys
import argparse
from typing import Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "attacks.db")
TARGET_EMAIL = "perezllorenc@gmail.com"


def get_db_path() -> str:
    return os.environ.get("ATTACKS_DB_PATH", DEFAULT_DB_PATH)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def list_users(db_path: Optional[str] = None) -> list[dict]:
    """Lista todos los usuarios registrados."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id, email, name, created_at FROM users ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_user(email: str, db_path: Optional[str] = None) -> dict:
    """
    Elimina un usuario por su email y todos sus ataques asociados.

    Returns:
        dict con 'ok', 'deleted_user' y 'deleted_attacks', o 'ok': False con 'error'.
    """
    conn = get_connection(db_path)
    try:
        user = conn.execute("SELECT id, name FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            return {"ok": False, "error": f"No se encontro usuario con email: {email}"}

        attacks_deleted = conn.execute(
            "DELETE FROM attacks WHERE user_email = ?", (email,)
        ).rowcount

        conn.execute("DELETE FROM users WHERE email = ?", (email,))
        conn.commit()

        print(f"[OK] Usuario eliminado: {user['name']} <{email}>")
        print(f"[OK] {attacks_deleted} ataque(s) eliminados")

        return {
            "ok": True,
            "deleted_user": {"id": user["id"], "email": email, "name": user["name"]},
            "deleted_attacks": attacks_deleted,
        }
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {e}")
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Eliminar un usuario de la base de datos")
    parser.add_argument("email", nargs="?", default=TARGET_EMAIL, help="Email del usuario a eliminar")
    parser.add_argument("--list", action="store_true", help="Listar todos los usuarios")
    parser.add_argument("--db", default=None, help="Ruta alternativa a la base de datos")
    args = parser.parse_args()

    if args.list:
        users = list_users(args.db)
        if not users:
            print("No hay usuarios registrados.")
            return
        print(f"{'ID':<5} {'Email':<35} {'Nombre':<20} {'Creado'}")
        print("-" * 80)
        for u in users:
            print(f"{u['id']:<5} {u['email']:<35} {u['name']:<20} {u['created_at']}")
        return

    confirm = input(f"¿Eliminar usuario '{args.email}' y todos sus datos? [s/N] ")
    if confirm.lower() != "s":
        print("Cancelado.")
        return

    result = delete_user(args.email, args.db)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()