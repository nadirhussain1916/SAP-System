import pandas as pd
import psycopg2
from datetime import datetime

# Load cleaned Excel sheet
df = pd.read_excel("cleaned_sap_logs.xlsx")

# Preprocess columns
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Convert 'started' to time
df['started'] = pd.to_datetime(df['started'], errors='coerce').dt.time

# Convert work_progress_in_privilege_mode to boolean
df['work_progress_in_privilege_mode'] = df['work_progress_in_privilege_mode'].astype(str).str.lower().map({
    'true': True, '1': True, 'yes': True,
    'false': False, '0': False, 'no': False
}).fillna(False)

# Convert all other columns to integer (except known text fields)
text_columns = ['server', 'program', 'user']
for col in df.columns:
    if col not in text_columns + ['started', 'work_progress_in_privilege_mode']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="sap_logs",
    user="postgres",
    password="1234",
    host="localhost",
    port=5432
)
cursor = conn.cursor()

# Build INSERT statement
columns = ', '.join(f'"{col}"' if col == "user" else col for col in df.columns)
placeholders = ', '.join(['%s'] * len(df.columns))
insert_sql = f'INSERT INTO logs_fixed ({columns}) VALUES ({placeholders})'

# Insert each row
for row in df.itertuples(index=False, name=None):
    cursor.execute(insert_sql, row)

# Commit and close
conn.commit()
cursor.close()
conn.close()

print("Data inserted successfully into logs_fixed.")
