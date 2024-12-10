import os
import re
import gc
import pandas as pd
import sqlite3
import numpy as np
import torch
import faiss
import nltk
from tqdm import tqdm

# Enhanced machine learning and NLP libraries
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# GPU-accelerated clustering libraries
import cupy as cp
import cudf
import cuml
from cuml.cluster import KMeans as cuKMeans
from cuml.preprocessing import StandardScaler as cuStandardScaler

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ClusteringConfig:
    DB_PATH = "industry_data.db"
    INPUT_TABLE = "industry_categories"
    OUTPUT_TABLE = "categorized_companies"
    
    # High-performance embedding models
    EMBEDDING_MODELS = [
        "all-mpnet-base-v2",         
        "multi-qa-mpnet-base-dot-v1",
        "paraphrase-multilingual-MiniLM-L12-v2"
    ]
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBEDDINGS_FILE = "large_scale_embeddings.pt"
    
    # Optimized for large-scale processing
    MAX_BATCH_SIZE = 100000  # Matches 3090's memory capacity
    EMBEDDING_BATCH_SIZE = 1024
    
    CUSTOM_STOPWORDS = set(stopwords.words('english')).union({
        'and', 'of', 'ltd', 'pte', 'services', 'trade', 'products', 
        'goods', 'including', 'activities', 'other', 'sale', 
        'wholesale', 'retail', 'nec', 'online', 'excluding', 
        'product', 'dominant', 'without', 'trading', 'variety', 
        'eg', 'co', 'singapore'
    })

class GPUEmbeddingGenerator:
    @classmethod
    def generate_embeddings(cls, texts, model_name=None, batch_size=1024):
        """
        GPU-accelerated embedding generation with multiple model support
        
        Args:
            texts (list): List of preprocessed texts
            model_name (str, optional): Specific model to use
            batch_size (int): Batch size for processing
        
        Returns:
            numpy.ndarray: Generated embeddings
        """
        if model_name is None:
            model_name = ClusteringConfig.EMBEDDING_MODELS[0]
        
        print(f"🚀 Generating GPU embeddings using {model_name}")
        model = SentenceTransformer(model_name).to('cuda')
        model.eval()  # Set to evaluation mode
        
        # Preallocate GPU memory for embeddings
        total_texts = len(texts)
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, total_texts, batch_size), desc="Embedding Generation"):
                batch_texts = texts[i:i+batch_size]
                
                # Batch processing with GPU acceleration
                batch_embeddings = model.encode(
                    batch_texts, 
                    batch_size=batch_size, 
                    show_progress_bar=False, 
                    device='cuda', 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                all_embeddings.append(batch_embeddings)
                
                # Free up GPU memory
                torch.cuda.empty_cache()
        
        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        print(f"🔍 Generated embeddings shape: {final_embeddings.shape}")
        return final_embeddings

class AdvancedTextProcessor:
    @staticmethod
    def preprocess_text(text, lemmatizer=None):
        """
        Enhanced text preprocessing with GPU-friendly preprocessing
        """
        if lemmatizer is None:
            lemmatizer = WordNetLemmatizer()
        
        # Comprehensive text cleaning
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        
        words = text.split()
        
        cleaned_words = [
            lemmatizer.lemmatize(word, pos='v')  # Verb-based lemmatization
            for word in words 
            if (word not in ClusteringConfig.CUSTOM_STOPWORDS) 
            and len(word) > 2
        ]
        
        return ' '.join(cleaned_words)

def gpu_kmeans_clustering(embeddings, n_clusters=50):
    """
    GPU-accelerated K-Means clustering using CUDA
    
    Args:
        embeddings (numpy.ndarray): Input embeddings
        n_clusters (int): Number of clusters
    
    Returns:
        numpy.ndarray: Cluster labels
    """
    # Convert to cupy array for GPU processing
    gpu_embeddings = cp.asarray(embeddings)
    
    # GPU-based StandardScaler
    scaler = cuStandardScaler()
    scaled_embeddings = scaler.fit_transform(gpu_embeddings)
    
    # GPU K-Means
    kmeans = cuKMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=10,
        max_iter=300
    )
    
    # Fit and predict
    labels = kmeans.fit_predict(scaled_embeddings)
    
    # Transfer back to CPU
    return cp.asnumpy(labels)

def faiss_clustering(embeddings, n_clusters=50):
    """
    FAISS-based GPU clustering for large-scale datasets
    
    Args:
        embeddings (numpy.ndarray): Input embeddings
        n_clusters (int): Number of clusters
    
    Returns:
        numpy.ndarray: Cluster labels
    """
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    # Create FAISS index
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    
    # If GPU is available, use GPU index
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Perform clustering
    kmeans = faiss.Clustering(d, n_clusters)
    kmeans.verbose = True
    kmeans.train(embeddings.astype('float32'), index)
    
    # Get cluster assignments
    _, labels = index.search(embeddings.astype('float32'), 1)
    
    return labels.flatten()

def main():
    # Database connection
    conn = sqlite3.connect(ClusteringConfig.DB_PATH)
    
    # Load data with cuDF for GPU-accelerated processing
    df = cudf.read_sql(
        f"SELECT entity_name, primary_ssic_description FROM {ClusteringConfig.INPUT_TABLE}", 
        conn
    )
    conn.close()
    
    # Intelligent sampling for massive datasets
    if len(df) > ClusteringConfig.MAX_BATCH_SIZE:
        df = df.sample(n=ClusteringConfig.MAX_BATCH_SIZE, random_state=42)
    
    # Preprocessing
    df['processed_text'] = df['entity_name'] + ' ' + df['primary_ssic_description']
    df['processed_text'] = df['processed_text'].apply(AdvancedTextProcessor.preprocess_text)
    
    # Generate embeddings across multiple models
    all_embeddings = []
    for model_name in ClusteringConfig.EMBEDDING_MODELS:
        model_embeddings = GPUEmbeddingGenerator.generate_embeddings(
            df['processed_text'].tolist(), 
            model_name,
            batch_size=ClusteringConfig.EMBEDDING_BATCH_SIZE
        )
        all_embeddings.append(model_embeddings)
    
    # Average embeddings from multiple models
    final_embeddings = np.mean(all_embeddings, axis=0)
    
    # Multiple clustering approaches
    clustering_methods = [
        ('GPU K-Means', gpu_kmeans_clustering),
        ('FAISS Clustering', faiss_clustering)
    ]
    
    best_labels = None
    best_score = -np.inf
    
    for method_name, clustering_func in clustering_methods:
        try:
            # Perform clustering
            labels = clustering_func(final_embeddings)
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            score = silhouette_score(final_embeddings, labels, metric='cosine')
            
            print(f"{method_name} Silhouette Score: {score}")
            
            if score > best_score:
                best_score = score
                best_labels = labels
        except Exception as e:
            print(f"Error with {method_name}: {e}")
    
    # Assign labels to dataframe
    df['cluster'] = best_labels
    
    # Perform text-based clustering keyword extraction
    # (You can keep your existing assign_category_keywords function here)
    
    # Save results
    df.to_sql(
        ClusteringConfig.OUTPUT_TABLE, 
        conn, 
        if_exists='replace', 
        index=False
    )
    
    print("🎉 Clustering Complete!")

if __name__ == "__main__":
    main()