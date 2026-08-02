"""Modulo de base de datos."""
import sqlite3
import os
import logging
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext
import jose.jwt

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "database")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "attacks.db")
_jwt_secret_file = os.path.join(DEFAULT_DB_DIR, ".jwt_secret")

JWT_SECRET = os.environ.get("JWT_SECRET_KEY")
if not JWT_SECRET:
    try:
        JWT_SECRET = Path(_jwt_secret_file).read_text().strip()
    except FileNotFoundError:
        JWT_SECRET = secrets.token_hex(32)
        os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
        Path(_jwt_secret_file).write_text(JWT_SECRET)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated=["auto"])


def get_db_path() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_PATH)


def get_connection(db_path=None):
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path=None):
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS attacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            final_verdict TEXT NOT NULL CHECK(final_verdict IN ('CLEAN','BLOCKED')),
            blocked_at_layer INTEGER,
            layer1_score REAL,
            layer1_suspicious INTEGER DEFAULT 0,
            layer1_categories TEXT,
            layer2_label TEXT,
            layer2_confidence REAL,
            layer2_score REAL,
            layer3_is_good INTEGER,
            layer3_score REAL,
            layer3_evaluation TEXT,
            processing_time REAL,
            source_ip TEXT,
            user_agent TEXT,
            user_email TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_attacks_verdict ON attacks(final_verdict);
        CREATE INDEX IF NOT EXISTS idx_attacks_created ON attacks(created_at);
        CREATE INDEX IF NOT EXISTS idx_attacks_source_ip ON attacks(source_ip);
        CREATE INDEX IF NOT EXISTS idx_attacks_blocked ON attacks(blocked_at_layer);
        CREATE INDEX IF NOT EXISTS idx_attacks_user_email ON attacks(user_email);
    """)
    conn.commit()
    conn.close()


def _hash_password(password):
    return pwd_context.hash(password)


def _verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def register_user(email, password, name):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return {"ok": False, "error": "Email ya registrado"}
        hashed_password = _hash_password(password)
        cursor.execute("INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                      (email, hashed_password, name))
        conn.commit()
        return {"ok": True, "user": {"email": email, "name": name}}
    except Exception as e:
        conn.rollback()
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def authenticate_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT email, name, password_hash FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if user and _verify_password(password, user["password_hash"]):
            return {"email": user["email"], "name": user["name"]}
        return None
    finally:
        conn.close()


def create_token(user, expires_delta=None):
    to_encode = user.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    return jose.jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token):
    try:
        return jose.jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jose.ExpiredSignatureError, jose.JWTError):
        return None


def log_attack(result, source_ip, user_agent, user_email=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        l1 = result.get('layer1', {})
        l2 = result.get('layer2', {})
        l3 = result.get('layer3', {})
        query = """
            INSERT INTO attacks (prompt, final_verdict, blocked_at_layer,
                layer1_score, layer1_suspicious, layer1_categories,
                layer2_label, layer2_confidence, layer2_score,
                layer3_is_good, layer3_score, layer3_evaluation,
                processing_time, source_ip, user_agent, user_email)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            result.get('prompt'), result.get('final_verdict'), result.get('blocked_at_layer'),
            l1.get('risk_score'), 1 if l1.get('is_suspicious') else 0,
            json.dumps(l1.get('triggered_categories', [])),
            l2.get('label'), l2.get('confidence'), l2.get('score'),
            1 if l3.get('is_good') else 0, l3.get('score'), l3.get('evaluation'),
            result.get('processing_time'), source_ip, user_agent, user_email
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
    finally:
        conn.close()


def get_dashboard_stats(user_email=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        where = "WHERE user_email = ?" if user_email else ""
        params = (user_email,) if user_email else ()
        cursor.execute(f"SELECT COUNT(*) as total FROM attacks {where}", params)
        total = cursor.fetchone()["total"]
        cursor.execute(f"SELECT COUNT(*) as blocked FROM attacks WHERE final_verdict = 'BLOCKED' {where}", params)
        blocked = cursor.fetchone()["blocked"]
        cursor.execute(f"SELECT AVG(processing_time) as avg_time FROM attacks {where}", params)
        avg_time = cursor.fetchone()["avg_time"] or 0
        return {
            "total_attacks": total, "blocked_attacks": blocked,
            "clean_attacks": total - blocked,
            "block_rate": round(blocked / total * 100, 2) if total > 0 else 0,
            "avg_processing_time": round(avg_time, 4)
        }
    finally:
        conn.close()


def get_recent_attacks(user_email=None, limit=50):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        where = "WHERE user_email = ?" if user_email else ""
        params = (user_email,) if user_email else ()
        q = f"SELECT id, prompt, final_verdict, blocked_at_layer, processing_time, created_at FROM attacks {where} ORDER BY created_at DESC LIMIT ?"
        cursor.execute(q, params + (limit,))
        return [{
            "id": r["id"], "prompt": r["prompt"][:200] + "..." if len(r["prompt"]) > 200 else r["prompt"],
            "final_verdict": r["final_verdict"], "blocked_at_layer": r["blocked_at_layer"],
            "processing_time": r["processing_time"], "created_at": r["created_at"]
        } for r in cursor.fetchall()]
    finally:
        conn.close()


def get_top_source_ips(user_email=None, limit=10):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        where = "WHERE user_email = ?" if user_email else ""
        params = (user_email,) if user_email else ()
        q = f"SELECT source_ip, COUNT(*) as count, SUM(CASE WHEN final_verdict = 'BLOCKED' THEN 1 ELSE 0 END) as blocked FROM attacks {where} GROUP BY source_ip ORDER BY count DESC LIMIT ?"
        cursor.execute(q, params + (limit,))
        return [{"ip": r["source_ip"] or "unknown", "total": r["count"], "blocked": r["blocked"]} for r in cursor.fetchall()]
    finally:
        conn.close()


def get_category_stats(user_email=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        where = "WHERE user_email = ?" if user_email else ""
        params = (user_email,) if user_email else ()
        cursor.execute(f"SELECT layer1_categories FROM attacks {where}", params)
        counts = {}
        for row in cursor.fetchall():
            for cat in json.loads(row["layer1_categories"] or "[]"):
                counts[cat] = counts.get(cat, 0) + 1
        return [{"category": c, "count": n} for c, n in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
    finally:
        conn.close()


def get_layer_detection_stats(user_email=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        where = "WHERE user_email = ?" if user_email else ""
        params = (user_email,) if user_email else ()
        cursor.execute(f"SELECT COUNT(*) as total FROM attacks {where}", params)
        total = cursor.fetchone()["total"]
        cursor.execute(f"SELECT COUNT(*) as layer1 FROM attacks WHERE layer1_suspicious = 1 {where}", params)
        l1 = cursor.fetchone()["layer1"]
        cursor.execute(f"SELECT COUNT(*) as layer2 FROM attacks WHERE layer2_label = 'injection' {where}", params)
        l2 = cursor.fetchone()["layer2"]
        return {
            "total": total, "layer1_detections": l1, "layer2_detections": l2,
            "layer1_rate": round(l1 / total * 100, 2) if total > 0 else 0,
            "layer2_rate": round(l2 / total * 100, 2) if total > 0 else 0
        }
    finally:
        conn.close()


def clear_attacks(user_email):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM attacks WHERE user_email = ?", (user_email,))
        conn.commit()
    finally:
        conn.close()


def get_attacks_timeline(user_email=None, days=30):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        where = "WHERE user_email = ?" if user_email else ""
        params = (user_email,) if user_email else ()
        q = f"SELECT DATE(created_at) as date, COUNT(*) as count, SUM(CASE WHEN final_verdict = 'BLOCKED' THEN 1 ELSE 0 END) as blocked FROM attacks {where} AND created_at >= datetime('now', ?) GROUP BY DATE(created_at) ORDER BY date"
        cursor.execute(q, params + (f"-{days} days",))
        return [{"date": r["date"], "total": r["count"], "blocked": r["blocked"]} for r in cursor.fetchall()]
    finally:
        conn.close()
