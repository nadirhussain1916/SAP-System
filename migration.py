import psycopg2
import subprocess
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
SQL_FILE = "sap_logs.sql"

# Step 1: Connect and create DB if not exists
conn = psycopg2.connect(
    dbname="postgres", user=DB_USER, password=DB_PASSWORD,
    host=DB_HOST, port=DB_PORT
)
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
exists = cur.fetchone()
if not exists:
    cur.execute(f"CREATE DATABASE {DB_NAME}")
    print(f"Database '{DB_NAME}' created.")
else:
    print(f"Database '{DB_NAME}' already exists.")
cur.close()
conn.close()

# Step 2: Import SQL dump
env = os.environ.copy()
env["PGPASSWORD"] = DB_PASSWORD

cmd = [
    "psql",
    "-h", DB_HOST,
    "-p", DB_PORT,
    "-U", DB_USER,
    "-d", DB_NAME,
    "-f", SQL_FILE
]

subprocess.run(cmd, env=env, check=True)
print(f"Imported '{SQL_FILE}' into database '{DB_NAME}'.")
