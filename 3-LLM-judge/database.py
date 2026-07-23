"""
Modulo de base de datos para el sistema de deteccion de Prompt Injection
========================================================================

Maneja la persistencia de ataques detectados usando SQLite.
Registra cada prompt analizado con su veredicto, fuente y metricas.
Incluye autenticacion de usuarios.
"""

import sqlite3
import os
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "attacks.db")

_db_dir = os.path.join(os.path.dirname(__file__), "..", "data")
_jwt_secret_file = os.path.join(_db_dir, ".jwt_secret")
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    try:
        JWT_SECRET = Path(_jwt_secret_file).read_text().strip()
    except FileNotFoundError:
        JWT_SECRET = secrets.token_hex(32)
        os.makedirs(_db_dir, exist_ok=True)
        Path(_jwt_secret_file).write_text(JWT_SECRET)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def get_db_path() -> str:
    return os.environ.get("ATTACKS_DB_PATH", DEFAULT_DB_PATH)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Optional[str] = None):
    conn = get_connection(db_path)

    # Migracion: anadir user_email a attacks si no existe (tabla preexistente)
    try:
        conn.execute("SELECT user_email FROM attacks LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE attacks ADD COLUMN user_email TEXT")
        logger.info("Columna user_email anadida a tabla attacks")
        conn.commit()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            email           TEXT    NOT NULL UNIQUE,
            password_hash   TEXT    NOT NULL,
            name            TEXT    NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS attacks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt          TEXT    NOT NULL,
            final_verdict   TEXT    NOT NULL CHECK(final_verdict IN ('CLEAN','BLOCKED')),
            blocked_at_layer INTEGER,
            layer1_score    REAL,
            layer1_suspicious INTEGER DEFAULT 0,
            layer1_categories TEXT,
            layer2_label    TEXT,
            layer2_confidence REAL,
            layer2_score    REAL,
            layer3_is_good  INTEGER,
            layer3_score    REAL,
            layer3_evaluation TEXT,
            processing_time REAL,
            source_ip       TEXT,
            user_agent      TEXT,
            user_email      TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_attacks_verdict   ON attacks(final_verdict);
        CREATE INDEX IF NOT EXISTS idx_attacks_created   ON attacks(created_at);
        CREATE INDEX IF NOT EXISTS idx_attacks_source_ip ON attacks(source_ip);
        CREATE INDEX IF NOT EXISTS idx_attacks_blocked   ON attacks(blocked_at_layer);
        CREATE INDEX IF NOT EXISTS idx_attacks_user_email ON attacks(user_email);
    """)
    conn.commit()
    conn.close()
    logger.info(f"Base de datos inicializada en: {db_path or get_db_path()}")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, password_hash: str) -> bool:
    salt, h = password_hash.split(":")
    return hashlib.sha256((salt + password).encode()).hexdigest() == h


def register_user(email: str, password: str, name: str,
                  db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            return {"ok": False, "error": "El email ya esta registrado"}

        pw_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
            (email, pw_hash, name),
        )
        conn.commit()
        logger.info(f"Usuario registrado: {email}")
        return {"ok": True}
    finally:
        conn.close()


def authenticate_user(email: str, password: str,
                      db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, email, name, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "email": row["email"], "name": row["name"]}
    finally:
        conn.close()


def create_token(user: Dict[str, Any]) -> str:
    import jwt
    payload = {
        "sub": user["id"],
        "email": user["email"],
        "name": user["name"],
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    import jwt
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Attack logging (now with user_email)
# ---------------------------------------------------------------------------

def log_attack(result: Dict[str, Any], source_ip: Optional[str] = None,
               user_agent: Optional[str] = None, user_email: Optional[str] = None,
               db_path: Optional[str] = None):
    conn = get_connection(db_path)
    try:
        l1 = result.get("layer1", {})
        l2 = result.get("layer2", {})
        l3 = result.get("layer3", {})

        conn.execute("""
            INSERT INTO attacks (
                prompt, final_verdict, blocked_at_layer,
                layer1_score, layer1_suspicious, layer1_categories,
                layer2_label, layer2_confidence, layer2_score,
                layer3_is_good, layer3_score, layer3_evaluation,
                processing_time, source_ip, user_agent, user_email
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get("prompt", ""),
            result.get("final_verdict", "CLEAN"),
            result.get("blocked_at_layer"),
            l1.get("risk_score"),
            1 if l1.get("is_suspicious") else 0,
            ",".join(l1.get("triggered_categories", [])),
            l2.get("label"),
            l2.get("confidence"),
            l2.get("score"),
            1 if l3.get("is_good") else 0,
            l3.get("score"),
            l3.get("evaluation"),
            result.get("processing_time"),
            source_ip,
            user_agent,
            user_email,
        ))
        conn.commit()
        logger.info(f"Ataque registrado: verdict={result.get('final_verdict')}, ip={source_ip}, user={user_email}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dashboard queries — all accept optional user_email to filter per user
# ---------------------------------------------------------------------------

def get_dashboard_stats(user_email: Optional[str] = None,
                        db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        where = "WHERE user_email = ?" if user_email else ""
        params = (user_email,) if user_email else ()

        row = conn.execute(f"SELECT COUNT(*) as total FROM attacks {where}", params).fetchone()
        total = row["total"]

        row = conn.execute(
            f"SELECT COUNT(*) as blocked FROM attacks {where} {'AND' if user_email else 'WHERE'} final_verdict='BLOCKED'",
            params,
        ).fetchone()
        blocked = row["blocked"]

        row = conn.execute(
            f"SELECT COUNT(*) as clean FROM attacks {where} {'AND' if user_email else 'WHERE'} final_verdict='CLEAN'",
            params,
        ).fetchone()
        clean = row["clean"]

        row = conn.execute(
            f"SELECT AVG(processing_time) as avg_time FROM attacks {where}", params
        ).fetchone()
        avg_time = row["avg_time"] or 0

        return {
            "total_requests": total,
            "blocked_attacks": blocked,
            "clean_prompts": clean,
            "avg_processing_time": round(avg_time, 4),
            "block_rate": round(blocked / total * 100, 2) if total > 0 else 0,
        }
    finally:
        conn.close()


def get_recent_attacks(limit: int = 50, verdict: Optional[str] = None,
                       user_email: Optional[str] = None,
                       db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        conditions = []
        params: list = []
        if user_email:
            conditions.append("user_email = ?")
            params.append(user_email)
        if verdict:
            conditions.append("final_verdict = ?")
            params.append(verdict)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = conn.execute(
            f"SELECT * FROM attacks {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_attacks_timeline(days: int = 7, user_email: Optional[str] = None,
                         db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        if user_email:
            rows = conn.execute("""
                SELECT date(created_at) as day,
                       final_verdict,
                       COUNT(*) as count
                FROM attacks
                WHERE created_at >= ? AND user_email = ?
                GROUP BY day, final_verdict
                ORDER BY day
            """, (since, user_email)).fetchall()
        else:
            rows = conn.execute("""
                SELECT date(created_at) as day,
                       final_verdict,
                       COUNT(*) as count
                FROM attacks
                WHERE created_at >= ?
                GROUP BY day, final_verdict
                ORDER BY day
            """, (since,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_top_source_ips(limit: int = 10, user_email: Optional[str] = None,
                       db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        if user_email:
            rows = conn.execute("""
                SELECT source_ip,
                       COUNT(*) as total,
                       SUM(CASE WHEN final_verdict='BLOCKED' THEN 1 ELSE 0 END) as blocked
                FROM attacks
                WHERE source_ip IS NOT NULL AND source_ip != '' AND user_email = ?
                GROUP BY source_ip
                ORDER BY total DESC
                LIMIT ?
            """, (user_email, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT source_ip,
                       COUNT(*) as total,
                       SUM(CASE WHEN final_verdict='BLOCKED' THEN 1 ELSE 0 END) as blocked
                FROM attacks
                WHERE source_ip IS NOT NULL AND source_ip != ''
                GROUP BY source_ip
                ORDER BY total DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_category_stats(user_email: Optional[str] = None,
                       db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        if user_email:
            rows = conn.execute("""
                SELECT layer1_categories as category, COUNT(*) as count
                FROM attacks
                WHERE layer1_categories IS NOT NULL AND layer1_categories != ''
                  AND user_email = ?
                GROUP BY layer1_categories
                ORDER BY count DESC
            """, (user_email,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT layer1_categories as category, COUNT(*) as count
                FROM attacks
                WHERE layer1_categories IS NOT NULL AND layer1_categories != ''
                GROUP BY layer1_categories
                ORDER BY count DESC
            """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_layer_detection_stats(user_email: Optional[str] = None,
                              db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        if user_email:
            row = conn.execute("""
                SELECT
                    SUM(layer1_suspicious) as layer1_detected,
                    SUM(CASE WHEN layer2_label='injection' THEN 1 ELSE 0 END) as layer2_detected,
                    SUM(CASE WHEN layer3_is_good=0 THEN 1 ELSE 0 END) as layer3_detected,
                    COUNT(*) as total
                FROM attacks WHERE final_verdict='BLOCKED' AND user_email = ?
            """, (user_email,)).fetchone()
        else:
            row = conn.execute("""
                SELECT
                    SUM(layer1_suspicious) as layer1_detected,
                    SUM(CASE WHEN layer2_label='injection' THEN 1 ELSE 0 END) as layer2_detected,
                    SUM(CASE WHEN layer3_is_good=0 THEN 1 ELSE 0 END) as layer3_detected,
                    COUNT(*) as total
                FROM attacks WHERE final_verdict='BLOCKED'
            """).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def clear_attacks(user_email: Optional[str] = None, db_path: Optional[str] = None):
    conn = get_connection(db_path)
    try:
        if user_email:
            conn.execute("DELETE FROM attacks WHERE user_email = ?", (user_email,))
        else:
            conn.execute("DELETE FROM attacks")
        conn.commit()
        logger.info(f"Tabla de ataques limpiada (user={user_email or 'ALL'})")
    finally:
        conn.close()
