#!/usr/bin/env python3
"""
Interactive Query Interface for NLP Model Comparison
Allows you to input custom queries and see results from all trained models
"""

import sys
import numpy as np
import pandas as pd
from data_loader_local_only import DataLoader
from models import (
    TFIDFCosineSimilarity, StringMatching, BERTSemanticModel,
    Word2VecCosineSimilarity, RDFEmbeddingModel, WordMoversDistance,
    SentimentFeatureModel
)

class InteractiveQuerySystem:
    def __init__(self):
        self.models = {}
        self.documents = []
        self.document_metadata = []
        self.is_initialized = False
    
    def initialize_system(self):
        """Initialize and train all models"""
        print("🚀 Initializing Interactive Query System...")
        print("=" * 60)
        
        # Load and preprocess data
        print("📊 Loading data...")
        data_loader = DataLoader()
        
        # Load datasets
        tourism_data = data_loader.load_tourism_data()
        ecommerce_data = data_loader.load_ecommerce_data_sample()
        processed_data = data_loader.preprocess_data()
        
        if processed_data is None or len(processed_data) == 0:
            print("❌ Error: No data available. Cannot initialize system.")
            return False
        
        # Extract documents and metadata
        self.documents = processed_data['text'].tolist()
        self.document_metadata = processed_data[['query', 'source', 'relevance_score']].to_dict('records')
        
        print(f"✅ Loaded {len(self.documents)} documents")
        
        # Initialize and train models
        print("\n🤖 Training models...")
        model_classes = [
            TFIDFCosineSimilarity,
            StringMatching,
            BERTSemanticModel,
            Word2VecCosineSimilarity,
            RDFEmbeddingModel,
            WordMoversDistance,
            SentimentFeatureModel
        ]
        
        training_texts = processed_data['text'].tolist()
        training_labels = processed_data['label'].tolist()
        
        for model_class in model_classes:
            try:
                print(f"  Training {model_class.__name__}...")
                model = model_class()
                model.train(training_texts, training_labels)
                
                if model.is_trained:
                    self.models[model.name] = model
                    print(f"  ✅ {model.name} trained successfully")
                else:
                    print(f"  ❌ {model.name} failed to train")
                    
            except Exception as e:
                print(f"  ❌ Error training {model_class.__name__}: {e}")
        
        if len(self.models) == 0:
            print("❌ No models were successfully trained!")
            return False
        
        print(f"\n✅ Successfully trained {len(self.models)} models")
        self.is_initialized = True
        return True
    
    def query_models(self, query, top_k=5):
        """Query all models and return ranked results"""
        if not self.is_initialized:
            print("❌ System not initialized. Please run initialize_system() first.")
            return None
        
        print(f"\n🔍 Query: '{query}'")
        print("=" * 60)
        
        all_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Get similarity scores for all documents
                similarities = model.get_similarity_scores(query, self.documents)
                
                if len(similarities) == 0:
                    print(f"⚠️  {model_name}: No results returned")
                    continue
                
                # Get top-k results
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for i, idx in enumerate(top_indices):
                    results.append({
                        'rank': i + 1,
                        'document': self.documents[idx][:100] + "..." if len(self.documents[idx]) > 100 else self.documents[idx],
                        'full_document': self.documents[idx],
                        'similarity_score': similarities[idx],
                        'source': self.document_metadata[idx]['source'],
                        'original_query': self.document_metadata[idx]['query'],
                        'relevance_score': self.document_metadata[idx]['relevance_score']
                    })
                
                all_results[model_name] = results
                
            except Exception as e:
                print(f"❌ Error querying {model_name}: {e}")
        
        return all_results
    
    def display_results(self, results, show_details=False):
        """Display query results in a formatted way"""
        if not results:
            print("No results to display.")
            return
        
        for model_name, model_results in results.items():
            print(f"\n📋 {model_name}")
            print("-" * 50)
            
            if not model_results:
                print("  No results returned")
                continue
            
            for result in model_results:
                print(f"  {result['rank']}. Score: {result['similarity_score']:.4f}")
                print(f"     {result['document']}")
                
                if show_details:
                    print(f"     Source: {result['source']} | Original Query: {result['original_query']}")
                    print(f"     Ground Truth Relevance: {result['relevance_score']:.2f}")
                print()
    
    def compare_models_for_query(self, query, top_k=5):
        """Compare all models for a single query with detailed analysis"""
        results = self.query_models(query, top_k)
        
        if not results:
            return
        
        print(f"\n📊 MODEL COMPARISON FOR QUERY: '{query}'")
        print("=" * 80)
        
        # Create comparison table
        comparison_data = []
        
        for model_name, model_results in results.items():
            if model_results:
                avg_similarity = np.mean([r['similarity_score'] for r in model_results])
                avg_relevance = np.mean([r['relevance_score'] for r in model_results])
                top_1_similarity = model_results[0]['similarity_score']
                top_1_relevance = model_results[0]['relevance_score']
                
                comparison_data.append({
                    'Model': model_name,
                    'Avg Similarity': avg_similarity,
                    'Top-1 Similarity': top_1_similarity,
                    'Avg Ground Truth Relevance': avg_relevance,
                    'Top-1 Ground Truth Relevance': top_1_relevance
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values('Top-1 Similarity', ascending=False)
            print("\nModel Performance Summary:")
            print(df.to_string(index=False, float_format='%.4f'))
        
        # Show detailed results
        print(f"\n📋 DETAILED RESULTS (Top {top_k})")
        print("=" * 80)
        self.display_results(results, show_details=True)
    
    def interactive_mode(self):
        """Run interactive query mode"""
        if not self.is_initialized:
            if not self.initialize_system():
                return
        
        print("\n🎯 INTERACTIVE QUERY MODE")
        print("=" * 60)
        print("Enter your queries to see how different models rank documents.")
        print("Commands:")
        print("  - Enter any text to search")
        print("  - 'help' - Show this help message")
        print("  - 'models' - List available models")
        print("  - 'stats' - Show system statistics")
        print("  - 'quit' or 'exit' - Exit the system")
        print()
        
        while True:
            try:
                query = input("🔍 Enter your query (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  - Any text: Search for documents")
                    print("  - 'models': List trained models")
                    print("  - 'stats': Show system statistics")
                    print("  - 'quit': Exit the system")
                    continue
                
                elif query.lower() == 'models':
                    print(f"\n🤖 Available Models ({len(self.models)}):")
                    for i, model_name in enumerate(self.models.keys(), 1):
                        print(f"  {i}. {model_name}")
                    continue
                
                elif query.lower() == 'stats':
                    print(f"\n📊 System Statistics:")
                    print(f"  Documents: {len(self.documents)}")
                    print(f"  Trained Models: {len(self.models)}")
                    print(f"  Document Sources: {set(meta['source'] for meta in self.document_metadata)}")
                    continue
                
                elif len(query.strip()) == 0:
                    print("Please enter a valid query.")
                    continue
                
                # Process the query
                print(f"\n⏳ Processing query: '{query}'...")
                self.compare_models_for_query(query, top_k=5)
                
                # Ask if user wants to see more results
                while True:
                    more = input("\n❓ Show more details? (y/n): ").strip().lower()
                    if more in ['y', 'yes']:
                        results = self.query_models(query, top_k=10)
                        self.display_results(results, show_details=True)
                        break
                    elif more in ['n', 'no', '']:
                        break
                    else:
                        print("Please enter 'y' or 'n'")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function to run the interactive query system"""
    print("🔍 NLP MODEL COMPARISON - INTERACTIVE QUERY SYSTEM")
    print("=" * 60)
    
    system = InteractiveQuerySystem()
    
    # Check if user wants to initialize or go straight to interactive mode
    print("Choose an option:")
    print("1. Initialize system and enter interactive mode")
    print("2. Test with sample queries")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            system.interactive_mode()
            break
        elif choice == '2':
            if system.initialize_system():
                test_sample_queries(system)
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3")

def test_sample_queries(system):
    """Test the system with sample queries"""
    sample_queries = [
        "best smartphone for students",
        "comfortable running shoes",
        "laptop for creative work",
        "beach vacation packages",
        "fitness equipment for home",
        "affordable headphones",
        "tourism behavior analysis"
    ]
    
    print("\n🧪 TESTING WITH SAMPLE QUERIES")
    print("=" * 60)
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n📝 Sample Query {i}/{len(sample_queries)}")
        system.compare_models_for_query(query, top_k=3)
        
        if i < len(sample_queries):
            input("\nPress Enter to continue to next query...")

if __name__ == "__main__":
    main()
