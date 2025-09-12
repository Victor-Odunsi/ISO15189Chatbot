import sqlite3
from datetime import datetime

DB_NAME = 'ISO15189'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS application_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_question TEXT,
            gpt_answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

def insert_application_logs(session_id, user_question, gpt_answer):

    if hasattr(gpt_answer, "content"):
        gpt_answer = gpt_answer.content
    elif isinstance(gpt_answer, dict): 
        gpt_answer = str(gpt_answer.get("output"))
    elif not isinstance(gpt_answer, str): 
        gpt_answer = str(gpt_answer)

    # --- Insert ---
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO application_logs (session_id, user_question, gpt_answer)
        VALUES (?, ?, ?)
        """,
        (str(session_id), str(user_question), str(gpt_answer))  # enforce strings
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT user_question, gpt_answer 
        FROM application_logs 
        WHERE session_id = ?
        """,
        (session_id,) 
    )
    messages = []
    for row in rows.fetchall():
        messages.append({"role": "user", "content": row["user_question"]})
        messages.append({"role": "assistant", "content": row["gpt_answer"]})

    conn.close()
    return messages

create_application_logs()