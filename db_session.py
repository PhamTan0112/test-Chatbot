import psycopg2
from psycopg2.extras import RealDictCursor
import urllib.parse as urlparse

DATABASE_URL = "postgresql://neondb_owner:npg_2wFGSBfrD3km@ep-gentle-feather-a5qjkjgb-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

url = urlparse.urlparse(DATABASE_URL)
DB_CONFIG = {
    "dbname": url.path[1:],
    "user": url.username,
    "password": url.password,
    "host": url.hostname,
    "port": url.port,
    "sslmode": "require"
}


def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    ensure_table_exists(conn)
    return conn


def ensure_table_exists(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()


def append_session(user_id: str, question: str, answer: str, max_rows_per_user=100):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions (user_id, question, answer)
                    VALUES (%s, %s, %s)
                """, (user_id, question, answer))

                cur.execute("""
                    DELETE FROM chat_sessions
                    WHERE user_id = %s
                    AND id NOT IN (
                        SELECT id FROM chat_sessions
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    )
                """, (user_id, user_id, max_rows_per_user))
    except Exception as e:
        print("DB Error [append_session]:", e)


def get_recent_session(user_id: str, limit=10) -> str:
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT question, answer
                    FROM chat_sessions
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, limit))
                rows = cur.fetchall()
                return "\n".join([
                    f"Q: {r['question']}\nA: {r['answer']}" for r in reversed(rows)
                ])
    except Exception as e:
        print("DB Error [get_recent_session]:", e)
        return ""
