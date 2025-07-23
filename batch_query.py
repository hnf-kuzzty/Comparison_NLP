#!/usr/bin/env python3
"""
Batch Query Script - Test multiple queries and save results
"""

import json
import pandas as pd
from datetime import datetime
from interactive_query import InteractiveQuerySystem

def batch_query_test():
    """Run batch queries and save results"""
    
    # Test queries covering different scenarios
    test_queries = [
        # E-commerce queries
        "best smartphone under 500 dollars",
        "comfortable running shoes for daily exercise",
        "laptop computer for gaming and design work",
        "wireless headphones with noise cancellation",
        "kitchen appliances for small apartment",
        
        # Tourism queries
        "beach vacation packages all inclusive",
        "mountain hiking adventure tours",
        "city break weekend getaway deals",
        "family friendly resort destinations",
        "budget travel tips for students",
        
        # General queries
        "gift ideas for fitness enthusiast",
        "home office furniture ergonomic",
        "organic skincare products sensitive skin",
        "books about personal development",
        "healthy meal prep recipes",
        
        # Ambiguous/challenging queries
        "something red",
        "fast delivery",
        "good quality",
        "popular choice",
        "recommended option"
    ]
    
    print("🔄 BATCH QUERY TESTING")
    print("=" * 60)
    print(f"Testing {len(test_queries)} queries...")
    
    # Initialize system
    system = InteractiveQuerySystem()
    if not system.initialize_system():
        print("❌ Failed to initialize system")
        return
    
    # Run all queries
    all_results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}/{len(test_queries)}: '{query}'")
        results = system.query_models(query, top_k=5)
        
        if results:
            all_results[query] = results
            
            # Show quick summary
            best_model = None
            best_score = 0
            
            for model_name, model_results in results.items():
                if model_results:
                    top_score = model_results[0]['similarity_score']
                    if top_score > best_score:
                        best_score = top_score
                        best_model = model_name
            
            print(f"   🏆 Best: {best_model} (score: {best_score:.4f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    json_filename = f"batch_query_results_{timestamp}.json"
    with open(json_filename, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for query, models in all_results.items():
            json_results[query] = {}
            for model_name, results in models.items():
                json_results[query][model_name] = [
                    {
                        'rank': r['rank'],
                        'document': r['document'],
                        'similarity_score': float(r['similarity_score']),
                        'source': r['source'],
                        'relevance_score': float(r['relevance_score'])
                    }
                    for r in results
                ]
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {json_filename}")
    
    # Create summary CSV
    summary_data = []
    for query, models in all_results.items():
        for model_name, results in models.items():
            if results:
                summary_data.append({
                    'query': query,
                    'model': model_name,
                    'top_1_score': results[0]['similarity_score'],
                    'top_1_relevance': results[0]['relevance_score'],
                    'avg_score': sum(r['similarity_score'] for r in results) / len(results),
                    'avg_relevance': sum(r['relevance_score'] for r in results) / len(results)
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_filename = f"batch_query_summary_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"📊 Summary saved to: {csv_filename}")
        
        # Show top performing models
        print(f"\n🏆 TOP PERFORMING MODELS (by average top-1 score):")
        model_performance = df.groupby('model')['top_1_score'].mean().sort_values(ascending=False)
        for i, (model, score) in enumerate(model_performance.items(), 1):
            print(f"{i}. {model:<30} Avg Score: {score:.4f}")

if __name__ == "__main__":
    batch_query_test()
