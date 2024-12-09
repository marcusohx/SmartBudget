import os
import pandas as pd
import sqlite3

# Define the folder containing the CSV files
folder_path = '../Dataset'  # Replace with your folder path

# Connect to SQLite database
conn = sqlite3.connect('industry_data.db')
cursor = conn.cursor()

# Create main table with auto-incrementing primary key
cursor.execute('''
CREATE TABLE IF NOT EXISTS industry_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uen TEXT,
    entity_name TEXT,
    primary_ssic_code TEXT,
    primary_ssic_description TEXT,
    secondary_ssic_code TEXT,
    secondary_ssic_description TEXT,
    category TEXT
)
''')

# Create indexes for fast retrieval
cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON industry_categories (entity_name)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_primary_ssic_code ON industry_categories (primary_ssic_code)')

# Scan the folder for CSV files
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract relevant columns for the main table
        columns_to_keep = [
            'uen', 
            'entity_name', 
            'primary_ssic_code', 
            'primary_ssic_description', 
            'secondary_ssic_code', 
            'secondary_ssic_description'
        ]
        filtered_df = df[columns_to_keep]

        # Add a default "Uncategorized" category column
        filtered_df['category'] = 'Uncategorized'

        # Insert into the main table
        filtered_df.to_sql('industry_categories', conn, if_exists='append', index=False)

# Commit changes
conn.commit()

# Show sample data
print("Sample data from industry_categories:")
df_result_main = pd.read_sql('SELECT * FROM industry_categories LIMIT 10', conn)
print(df_result_main)

# Close the connection
conn.close()
