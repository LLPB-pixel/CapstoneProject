"""
Modulo de base de datos para el sistema de deteccion de Prompt Injection
========================================================================

Maneja la persistencia de ataques detectados usando SQLite.
Registra cada prompt analizado con su veredicto, fuente y metricas.
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "attacks.db")


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
    conn.executescript("""
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
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_attacks_verdict   ON attacks(final_verdict);
        CREATE INDEX IF NOT EXISTS idx_attacks_created   ON attacks(created_at);
        CREATE INDEX IF NOT EXISTS idx_attacks_source_ip ON attacks(source_ip);
        CREATE INDEX IF NOT EXISTS idx_attacks_blocked   ON attacks(blocked_at_layer);
    """)
    conn.commit()
    conn.close()
    logger.info(f"Base de datos inicializada en: {db_path or get_db_path()}")


def log_attack(result: Dict[str, Any], source_ip: Optional[str] = None,
               user_agent: Optional[str] = None, db_path: Optional[str] = None):
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
                processing_time, source_ip, user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))
        conn.commit()
        logger.info(f"Ataque registrado: verdict={result.get('final_verdict')}, ip={source_ip}")
    finally:
        conn.close()


def get_dashboard_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) as total FROM attacks").fetchone()
        total = row["total"]

        row = conn.execute("SELECT COUNT(*) as blocked FROM attacks WHERE final_verdict='BLOCKED'").fetchone()
        blocked = row["blocked"]

        row = conn.execute("SELECT COUNT(*) as clean FROM attacks WHERE final_verdict='CLEAN'").fetchone()
        clean = row["clean"]

        row = conn.execute("SELECT AVG(processing_time) as avg_time FROM attacks").fetchone()
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
                       db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        if verdict:
            rows = conn.execute(
                "SELECT * FROM attacks WHERE final_verdict=? ORDER BY created_at DESC LIMIT ?",
                (verdict, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM attacks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_attacks_by_hour(days: int = 7, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT strftime('%%Y-%%m-%%d %%H:00:00', created_at) as hour,
                   final_verdict,
                   COUNT(*) as count
            FROM attacks
            WHERE created_at >= ?
            GROUP BY hour, final_verdict
            ORDER BY hour
        """, (since,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_top_source_ips(limit: int = 10, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
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


def get_category_stats(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
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


def get_layer_detection_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
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


def get_attacks_timeline(days: int = 7, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
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
