import os
import pandas as pd
import sqlite3
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# === CONFIGURATION ===
DB_PATH = "industry_data.db"  # Path to SQLite database
TABLE_NAME = "industry_categories"  # Input table name
OUTPUT_TABLE_NAME = "categorized_companies"  # Output table name
N_CATEGORIES = 10  # Set the number of categories
EMBEDDINGS_FILE = "bert_embeddings.pt"  # File to save/load embeddings


# === FUNCTIONS ===
def load_data_from_db(db_path, table_name):
    """Load data from SQLite database."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT entity_name, primary_ssic_description FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def preprocess_data(df):
    """Clean and combine text features."""
    df.fillna('', inplace=True)
    df['combined_features'] = df['entity_name'] + ' ' + df['primary_ssic_description']
    return df


def get_bert_embeddings(texts, model, tokenizer, max_length=128, batch_size=32):
    """Convert text data into embeddings using BERT with batch processing."""
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT Embeddings"):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**tokens)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        embeddings.append(batch_embeddings)

        # Save partial results to disk
        torch.save(torch.cat(embeddings), EMBEDDINGS_FILE)
    
    return torch.cat(embeddings)


def perform_clustering(embeddings, n_clusters):
    """Cluster the data into a fixed number of categories."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings.numpy())
    return labels, kmeans


def evaluate_clustering(embeddings, labels):
    """Evaluate clustering performance."""
    score = silhouette_score(embeddings.numpy(), labels)
    print(f"Silhouette Score: {score}")


def save_to_db(df, db_path, table_name):
    """Save the DataFrame back to SQLite."""
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()


# === MAIN WORKFLOW ===
def main():
    # Step 1: Load and preprocess data
    print("Loading data...")
    data = load_data_from_db(DB_PATH, TABLE_NAME)

    # Process only a subset (5% of the dataset)
    data = data.sample(frac=0.05, random_state=42)
    print(f"Using a subset of {len(data)} rows from the dataset.")
    data = preprocess_data(data)

    # Step 2: Check for existing embeddings or generate new ones
    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading existing embeddings from disk...")
        embeddings = torch.load(EMBEDDINGS_FILE)
    else:
        print("Generating BERT embeddings...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()  # Set model to evaluation mode
        embeddings = get_bert_embeddings(data["combined_features"].tolist(), model, tokenizer)

    # Step 3: Perform clustering to assign categories
    print(f"Clustering data into {N_CATEGORIES} categories...")
    data["category"], kmeans = perform_clustering(embeddings, N_CATEGORIES)

    # Step 4: Evaluate clustering performance
    print("Evaluating clustering...")
    evaluate_clustering(embeddings, data["category"])

    # Step 5: Save clustered data to database
    print("Saving clustered data to database...")
    save_to_db(data, DB_PATH, OUTPUT_TABLE_NAME)

    print("Process completed!")


if __name__ == "__main__":
    main()
