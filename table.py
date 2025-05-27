import psycopg2
import pandas as pd

# PostgreSQL connection details
conn = psycopg2.connect(
    dbname="sap_logs",
    user="postgres",
    password="1234",
    host="localhost",
    port=5432
)
cursor = conn.cursor()

# Query to get column names and data types for table 'logs_fixed'
cursor.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'logs_fixed'
    ORDER BY ordinal_position;
""")

# Fetch and store results in a DataFrame
columns_info = cursor.fetchall()
schema_df = pd.DataFrame(columns_info, columns=['column_name', 'data_type'])

# Save to CSV
schema_csv_path = 'table_schema.csv'
schema_df.to_csv(schema_csv_path, index=False)

# Clean up
cursor.close()
conn.close()

schema_csv_path
