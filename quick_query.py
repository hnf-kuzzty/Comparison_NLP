#!/usr/bin/env python3
"""
Quick Query Script - Test a single query against all models
Usage: python quick_query.py "your query here"
"""

import sys
import argparse
from interactive_query import InteractiveQuerySystem

def quick_query(query_text, top_k=5, show_details=False):
    """Run a quick query without interactive mode"""
    print(f"🔍 Quick Query: '{query_text}'")
    print("=" * 60)
    
    # Initialize system
    system = InteractiveQuerySystem()
    print("⏳ Initializing system...")
    
    if not system.initialize_system():
        print("❌ Failed to initialize system")
        return
    
    # Run query
    print(f"\n🔍 Searching for: '{query_text}'")
    results = system.query_models(query_text, top_k)
    
    if results:
        print(f"\n📊 RESULTS (Top {top_k})")
        print("=" * 60)
        system.display_results(results, show_details=show_details)
        
        # Show comparison summary
        comparison_data = []
        for model_name, model_results in results.items():
            if model_results:
                avg_similarity = sum(r['similarity_score'] for r in model_results) / len(model_results)
                top_similarity = model_results[0]['similarity_score']
                comparison_data.append((model_name, top_similarity, avg_similarity))
        
        if comparison_data:
            comparison_data.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🏆 MODEL RANKING FOR '{query_text}':")
            print("-" * 50)
            for i, (model, top_sim, avg_sim) in enumerate(comparison_data, 1):
                print(f"{i}. {model:<30} Top: {top_sim:.4f} | Avg: {avg_sim:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Quick query against trained NLP models')
    parser.add_argument('query', help='Query text to search for')
    parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of top results to show (default: 5)')
    parser.add_argument('-d', '--details', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    quick_query(args.query, args.top_k, args.details)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_query.py 'your query here'")
        print("Example: python quick_query.py 'best smartphone under 500'")
        sys.exit(1)
    
    main()
