import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from difflib import SequenceMatcher
import multiprocessing
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class BaseModel:
    def __init__(self, name):
        self.name = name
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    def train(self, texts, labels=None):
        raise NotImplementedError
    
    def predict(self, query, candidates):
        raise NotImplementedError
    
    def get_similarity_scores(self, query, candidates):
        raise NotImplementedError

class TFIDFCosineSimilarity(BaseModel):
    def __init__(self):
        super().__init__("TF-IDF + Cosine Similarity")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.document_vectors = None
        self.documents = None
    
    def train(self, texts, labels=None):
        """Train TF-IDF vectorizer"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.documents = processed_texts
        self.document_vectors = self.vectorizer.fit_transform(processed_texts)
        self.is_trained = True
        print(f"{self.name} trained on {len(texts)} documents")
    
    def get_similarity_scores(self, query, candidates=None):
        """Get similarity scores for query against all documents"""
        if not self.is_trained:
            return []
        
        query_processed = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([query_processed])
        
        if candidates is None:
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        else:
            candidate_vectors = self.vectorizer.transform([self.preprocess_text(c) for c in candidates])
            similarities = cosine_similarity(query_vector, candidate_vectors).flatten()
        
        return similarities

class StringMatching(BaseModel):
    def __init__(self):
        super().__init__("String Matching")
        self.documents = None
    
    def train(self, texts, labels=None):
        """Store documents for string matching"""
        self.documents = [self.preprocess_text(text) for text in texts]
        self.is_trained = True
        print(f"{self.name} trained on {len(texts)} documents")
    
    def get_similarity_scores(self, query, candidates=None):
        """Calculate string similarity scores"""
        if not self.is_trained:
            return []
        
        query_processed = self.preprocess_text(query)
        
        if candidates is None:
            candidates = self.documents
        else:
            candidates = [self.preprocess_text(c) for c in candidates]
        
        similarities = []
        for candidate in candidates:
            similarity = SequenceMatcher(None, query_processed, candidate).ratio()
            similarities.append(similarity)
        
        return np.array(similarities)

class BERTSemanticModel(BaseModel):
    def __init__(self):
        super().__init__("BERT-based Semantic Parsing")
        self.model = None
        self.document_embeddings = None
        self.documents = None
    
    def train(self, texts, labels=None):
        """Initialize SentenceTransformer model and create document embeddings"""
        try:
            # Use SentenceTransformer instead of raw transformers
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            self.documents = processed_texts
            
            # Create embeddings for all documents using batch processing
            # SentenceTransformer handles batching internally
            print(f"Creating embeddings for {len(processed_texts)} documents...")
            self.document_embeddings = self.model.encode(
                processed_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            
            self.is_trained = True
            print(f"{self.name} trained on {len(texts)} documents")
            print(f"Embedding shape: {self.document_embeddings.shape}")
            
        except Exception as e:
            print(f"Error initializing SentenceTransformer model: {e}")
            print("Falling back to TF-IDF similarity")
            self.fallback_model = TFIDFCosineSimilarity()
            self.fallback_model.train(texts, labels)
    
    def _get_embeddings(self, texts):
        """Get sentence embeddings using SentenceTransformer"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Fallback to random embeddings if model fails
            return np.random.rand(len(texts), 384)
    
    def get_similarity_scores(self, query, candidates=None):
        """Get semantic similarity scores using cosine similarity"""
        if not self.is_trained:
            return []
        
        # Use fallback model if available
        if hasattr(self, 'fallback_model'):
            return self.fallback_model.get_similarity_scores(query, candidates)
        
        try:
            # Get query embedding
            query_processed = self.preprocess_text(query)
            query_embedding = self._get_embeddings([query_processed])[0]
            
            # Get target embeddings
            if candidates is None:
                target_embeddings = self.document_embeddings
            else:
                candidate_processed = [self.preprocess_text(c) for c in candidates]
                target_embeddings = self._get_embeddings(candidate_processed)
            
            # Calculate cosine similarity (optimized with normalized embeddings)
            # Since embeddings are normalized, dot product equals cosine similarity
            similarities = np.dot(target_embeddings, query_embedding)
            
            return similarities
            
        except Exception as e:
            print(f"Error calculating similarity scores: {e}")
            # Return random scores as fallback
            num_targets = len(candidates) if candidates else len(self.documents)
            return np.random.rand(num_targets)
    
    def find_most_similar(self, query, top_k=5):
        """Find the most similar documents to the query"""
        if not self.is_trained or not hasattr(self, 'documents'):
            return []
        
        similarities = self.get_similarity_scores(query)
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx],
                'index': idx
            })
        
        return results
    
    def batch_similarity(self, queries, candidates=None):
        """Calculate similarity scores for multiple queries at once"""
        if not self.is_trained:
            return []
        
        if hasattr(self, 'fallback_model'):
            # Handle batch processing with fallback model
            return [self.fallback_model.get_similarity_scores(q, candidates) for q in queries]
        
        try:
            # Process queries in batch
            processed_queries = [self.preprocess_text(q) for q in queries]
            query_embeddings = self._get_embeddings(processed_queries)
            
            # Get target embeddings
            if candidates is None:
                target_embeddings = self.document_embeddings
            else:
                candidate_processed = [self.preprocess_text(c) for c in candidates]
                target_embeddings = self._get_embeddings(candidate_processed)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(query_embeddings, target_embeddings.T)
            
            return similarity_matrix
            
        except Exception as e:
            print(f"Error in batch similarity calculation: {e}")
            num_targets = len(candidates) if candidates else len(self.documents)
            return [np.random.rand(num_targets) for _ in queries]
    
    def save_embeddings(self, filepath):
        """Save document embeddings to file"""
        if self.document_embeddings is not None:
            np.save(filepath, self.document_embeddings)
            print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath, documents):
        """Load pre-computed embeddings from file"""
        try:
            self.document_embeddings = np.load(filepath)
            self.documents = documents
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_trained = True
            print(f"Embeddings loaded from {filepath}")
            print(f"Loaded {len(self.document_embeddings)} embeddings")
        except Exception as e:
            print(f"Error loading embeddings: {e}")

class Word2VecCosineSimilarity(BaseModel):
    def __init__(self, vector_size=200, window=10, min_count=2, epochs=20):
        super().__init__("Word2Vec + Cosine Similarity")
        self.model = None
        self.documents = None
        self.document_vectors = None
        self.word_weights = None  # For TF-IDF weighting
        
        # Hyperparameters
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = multiprocessing.cpu_count()
    
    def _calculate_word_weights(self, tokenized_texts):
        """Calculate TF-IDF weights for words to improve document representation"""
        word_doc_count = defaultdict(int)
        total_docs = len(tokenized_texts)
        
        # Count document frequency for each word
        for tokens in tokenized_texts:
            unique_words = set(tokens)
            for word in unique_words:
                word_doc_count[word] += 1
        
        # Calculate IDF weights
        word_weights = {}
        for word, doc_count in word_doc_count.items():
            # IDF = log(total_docs / doc_freq)
            idf = np.log(total_docs / doc_count)
            word_weights[word] = idf
        
        return word_weights
    
    def _create_document_vector(self, tokens, use_tfidf=True):
        """Create document vector with optional TF-IDF weighting"""
        if not tokens:
            return np.zeros(self.vector_size)
        
        # Get vectors for words that exist in the model
        word_vectors = []
        weights = []
        
        # Calculate term frequencies if using TF-IDF
        if use_tfidf and self.word_weights:
            token_counts = defaultdict(int)
            for token in tokens:
                token_counts[token] += 1
            
            for word in tokens:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
                    # TF-IDF weight = (tf / max_tf) * idf
                    tf = token_counts[word] / max(token_counts.values())
                    idf = self.word_weights.get(word, 1.0)
                    weights.append(tf * idf)
        else:
            # Simple averaging
            for word in tokens:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
                    weights.append(1.0)
        
        if not word_vectors:
            return np.zeros(self.vector_size)
        
        # Weighted average of word vectors
        word_vectors = np.array(word_vectors)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        document_vector = np.average(word_vectors, axis=0, weights=weights)
        return document_vector
    
    def train(self, texts, labels=None, use_pretrained=None, use_tfidf=True):
        """Train Word2Vec model with enhanced options"""
        print(f"Training {self.name} on {len(texts)} documents...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.documents = processed_texts
        
        # Tokenize texts for Word2Vec
        tokenized_texts = [simple_preprocess(text) for text in processed_texts]
        
        # Filter out empty documents
        tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
        
        if not tokenized_texts:
            raise ValueError("No valid documents found after preprocessing")
        
        try:
            if use_pretrained:
                # Load pre-trained Word2Vec model and update with new vocabulary
                print(f"Loading pre-trained model: {use_pretrained}")
                self.model = Word2Vec.load(use_pretrained)
                self.model.build_vocab(tokenized_texts, update=True)
                self.model.train(tokenized_texts, total_examples=len(tokenized_texts), epochs=self.epochs)
            else:
                # Train new Word2Vec model with optimized parameters
                self.model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=self.workers,
                    epochs=self.epochs,
                    sg=1,  # Skip-gram (better for small datasets)
                    hs=0,  # Use negative sampling
                    negative=10,  # Negative sampling count
                    alpha=0.025,  # Initial learning rate
                    min_alpha=0.0001,  # Final learning rate
                    seed=42  # For reproducibility
                )
            
            # Calculate word weights for TF-IDF weighting
            if use_tfidf:
                self.word_weights = self._calculate_word_weights(tokenized_texts)
            
            # Create document vectors
            print("Creating document vectors...")
            self.document_vectors = []
            for tokens in tokenized_texts:
                doc_vector = self._create_document_vector(tokens, use_tfidf)
                self.document_vectors.append(doc_vector)
            
            self.document_vectors = np.array(self.document_vectors)
            
            # Normalize document vectors for faster cosine similarity
            self.document_vectors = normalize(self.document_vectors, norm='l2')
            
            # Store vocabulary info
            vocab_size = len(self.model.wv.key_to_index)
            coverage = self._calculate_vocabulary_coverage(tokenized_texts)
            
            self.is_trained = True
            print(f"{self.name} training completed!")
            print(f"- Vocabulary size: {vocab_size}")
            print(f"- Vector size: {self.vector_size}")
            print(f"- Vocabulary coverage: {coverage:.2%}")
            print(f"- Document vectors shape: {self.document_vectors.shape}")
            
        except Exception as e:
            print(f"Error training Word2Vec model: {e}")
            raise
    
    def _calculate_vocabulary_coverage(self, tokenized_texts):
        """Calculate what percentage of words are covered by the model"""
        total_words = 0
        covered_words = 0
        
        for tokens in tokenized_texts:
            for word in tokens:
                total_words += 1
                if word in self.model.wv:
                    covered_words += 1
        
        return covered_words / total_words if total_words > 0 else 0
    
    def get_similarity_scores(self, query, candidates=None, use_tfidf=True):
        """Get Word2Vec-based similarity scores with optimized computation"""
        if not self.is_trained:
            return []
        
        # Create query vector
        query_processed = self.preprocess_text(query)
        query_tokens = simple_preprocess(query_processed)
        query_vector = self._create_document_vector(query_tokens, use_tfidf)
        
        # Normalize query vector
        query_vector = normalize([query_vector], norm='l2')[0]
        
        # Get target vectors
        if candidates is None:
            target_vectors = self.document_vectors
        else:
            target_vectors = []
            for candidate in candidates:
                candidate_processed = self.preprocess_text(candidate)
                candidate_tokens = simple_preprocess(candidate_processed)
                candidate_vector = self._create_document_vector(candidate_tokens, use_tfidf)
                target_vectors.append(candidate_vector)
            
            target_vectors = np.array(target_vectors)
            target_vectors = normalize(target_vectors, norm='l2')
        
        # Calculate cosine similarities using vectorized operations
        similarities = np.dot(target_vectors, query_vector)
        
        # Handle NaN values (in case of zero vectors)
        similarities = np.nan_to_num(similarities, nan=0.0)
        
        return similarities
    
    def find_most_similar_words(self, word, top_k=10):
        """Find most similar words to a given word"""
        if not self.is_trained or word not in self.model.wv:
            return []
        
        try:
            similar_words = self.model.wv.most_similar(word, topn=top_k)
            return similar_words
        except:
            return []
    
    def get_word_vector(self, word):
        """Get vector representation of a word"""
        if not self.is_trained or word not in self.model.wv:
            return None
        
        return self.model.wv[word]
    
    def find_most_similar_documents(self, query, top_k=5, use_tfidf=True):
        """Find most similar documents to query"""
        if not self.is_trained:
            return []
        
        similarities = self.get_similarity_scores(query, use_tfidf=use_tfidf)
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            })
        
        return results
    
    def save_model(self, filepath):
        """Save the trained Word2Vec model"""
        if self.is_trained:
            self.model.save(filepath)
            # Also save document vectors and other metadata
            np.savez(f"{filepath}_metadata.npz", 
                    document_vectors=self.document_vectors,
                    documents=self.documents,
                    word_weights=self.word_weights if self.word_weights else {})
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained Word2Vec model"""
        try:
            self.model = Word2Vec.load(filepath)
            
            # Load metadata if available
            metadata = np.load(f"{filepath}_metadata.npz", allow_pickle=True)
            self.document_vectors = metadata['document_vectors']
            self.documents = metadata['documents'].tolist()
            self.word_weights = metadata['word_weights'].item() if 'word_weights' in metadata else None
            
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            print(f"Vocabulary size: {len(self.model.wv.key_to_index)}")
            print(f"Document vectors shape: {self.document_vectors.shape}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return "Model not trained yet"
        
        vocab_size = len(self.model.wv.key_to_index)
        return {
            'model_name': self.name,
            'vocabulary_size': vocab_size,
            'vector_size': self.vector_size,
            'window_size': self.window,
            'min_count': self.min_count,
            'epochs': self.epochs,
            'num_documents': len(self.documents) if self.documents else 0,
            'uses_tfidf_weighting': self.word_weights is not None
        }

class RDFEmbeddingModel(BaseModel):
    def __init__(self):
        super().__init__("Linked Open Data with RDF-based Embedding")
        self.entity_embeddings = {}
        self.documents = None
    
    def train(self, texts, labels=None):
        """Simulate RDF-based embeddings"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.documents = processed_texts
        
        # Extract entities (simplified - just unique words)
        all_words = set()
        for text in processed_texts:
            words = text.split()
            all_words.update(words)
        
        # Create random embeddings for entities (simulating RDF embeddings)
        embedding_dim = 128
        for word in all_words:
            self.entity_embeddings[word] = np.random.rand(embedding_dim)
        
        self.is_trained = True
        print(f"{self.name} trained on {len(texts)} documents with {len(all_words)} entities")
    
    def get_similarity_scores(self, query, candidates=None):
        """Get RDF-based similarity scores"""
        if not self.is_trained:
            return []
        
        query_processed = self.preprocess_text(query)
        query_words = query_processed.split()
        
        # Create query embedding by averaging entity embeddings
        query_embeddings = [self.entity_embeddings.get(word, np.zeros(128)) for word in query_words]
        if query_embeddings:
            query_vector = np.mean(query_embeddings, axis=0)
        else:
            query_vector = np.zeros(128)
        
        if candidates is None:
            candidates = self.documents
        
        similarities = []
        for candidate in candidates:
            candidate_processed = self.preprocess_text(candidate)
            candidate_words = candidate_processed.split()
            
            candidate_embeddings = [self.entity_embeddings.get(word, np.zeros(128)) for word in candidate_words]
            if candidate_embeddings:
                candidate_vector = np.mean(candidate_embeddings, axis=0)
            else:
                candidate_vector = np.zeros(128)
            
            # Calculate cosine similarity
            if np.linalg.norm(query_vector) == 0 or np.linalg.norm(candidate_vector) == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vector, candidate_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(candidate_vector)
                )
            similarities.append(similarity)
        
        return np.array(similarities)

class WordMoversDistance(BaseModel):
    def __init__(self):
        super().__init__("Word Mover's Distance for Short Essay Evaluation")
        self.model = None
        self.documents = None
    
    def train(self, texts, labels=None):
        """Train Word2Vec for WMD calculation"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.documents = processed_texts
        
        # Tokenize texts
        tokenized_texts = [simple_preprocess(text) for text in processed_texts]
        
        # Train Word2Vec model for WMD
        self.model = Word2Vec(sentences=tokenized_texts, vector_size=100, 
                             window=5, min_count=1, workers=4, epochs=10)
        
        self.is_trained = True
        print(f"{self.name} trained on {len(texts)} documents")
    
    def get_similarity_scores(self, query, candidates=None):
        """Get WMD-based similarity scores (simplified implementation)"""
        if not self.is_trained:
            return []
        
        query_processed = self.preprocess_text(query)
        query_tokens = simple_preprocess(query_processed)
        
        if candidates is None:
            candidates = self.documents
        
        similarities = []
        for candidate in candidates:
            candidate_processed = self.preprocess_text(candidate)
            candidate_tokens = simple_preprocess(candidate_processed)
            
            # Simplified WMD calculation using word overlap and similarity
            if not query_tokens or not candidate_tokens:
                similarity = 0
            else:
                # Calculate word-level similarities
                word_similarities = []
                for q_word in query_tokens:
                    if q_word in self.model.wv:
                        max_sim = 0
                        for c_word in candidate_tokens:
                            if c_word in self.model.wv:
                                sim = self.model.wv.similarity(q_word, c_word)
                                max_sim = max(max_sim, sim)
                        word_similarities.append(max_sim)
                
                similarity = np.mean(word_similarities) if word_similarities else 0
            
            similarities.append(similarity)
        
        return np.array(similarities)

class SentimentFeatureModel(BaseModel):
    def __init__(self):
        super().__init__("Sentiment Feature Modeling Based on Complaints")
        self.sentiment_lexicon = None
        self.documents = None
        self.tfidf_vectorizer = None
        self.document_vectors = None
    
    def train(self, texts, labels=None):
        """Train sentiment-based feature model"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.documents = processed_texts
        
        # Create simple sentiment lexicon
        positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'best', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor']
        
        self.sentiment_lexicon = {}
        for word in positive_words:
            self.sentiment_lexicon[word] = 1
        for word in negative_words:
            self.sentiment_lexicon[word] = -1
        
        # Create TF-IDF features combined with sentiment features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Add sentiment features
        sentiment_features = []
        for text in processed_texts:
            words = text.split()
            sentiment_score = sum(self.sentiment_lexicon.get(word, 0) for word in words)
            sentiment_features.append([sentiment_score, len(words)])
        
        sentiment_features = np.array(sentiment_features)
        
        # Combine TF-IDF and sentiment features
        from scipy.sparse import hstack
        self.document_vectors = hstack([tfidf_features, sentiment_features])
        
        self.is_trained = True
        print(f"{self.name} trained on {len(texts)} documents")
    
    def get_similarity_scores(self, query, candidates=None):
        """Get sentiment-aware similarity scores"""
        if not self.is_trained:
            return []
        
        query_processed = self.preprocess_text(query)
        
        # Create query features
        query_tfidf = self.tfidf_vectorizer.transform([query_processed])
        query_words = query_processed.split()
        query_sentiment = sum(self.sentiment_lexicon.get(word, 0) for word in query_words)
        query_sentiment_features = np.array([[query_sentiment, len(query_words)]])
        
        from scipy.sparse import hstack
        query_vector = hstack([query_tfidf, query_sentiment_features])
        
        if candidates is None:
            target_vectors = self.document_vectors
        else:
            candidate_tfidf = self.tfidf_vectorizer.transform([self.preprocess_text(c) for c in candidates])
            candidate_sentiment_features = []
            for candidate in candidates:
                candidate_processed = self.preprocess_text(candidate)
                candidate_words = candidate_processed.split()
                sentiment_score = sum(self.sentiment_lexicon.get(word, 0) for word in candidate_words)
                candidate_sentiment_features.append([sentiment_score, len(candidate_words)])
            
            candidate_sentiment_features = np.array(candidate_sentiment_features)
            target_vectors = hstack([candidate_tfidf, candidate_sentiment_features])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, target_vectors).flatten()
        return similarities
