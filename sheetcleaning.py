import pandas as pd
import re

# Load the cleaned Excel file
file_path = 'cleaned_sap_logs.xlsx'
df = pd.read_excel(file_path)

# Function to clean numeric-looking strings with commas
def clean_numeric_commas(value):
    if isinstance(value, str):
        # Remove commas if the value looks like a number
        numeric_like = re.fullmatch(r'[\d,]+', value)
        if numeric_like:
            return value.replace(',', '')
    return value

# Apply the function to all cells
df_cleaned = df.applymap(clean_numeric_commas)

# Save the cleaned DataFrame to a new Excel file
output_path = 'cleaned_no_commas_sap_logs.xlsx'
df_cleaned.to_excel(output_path, index=False)

output_path
