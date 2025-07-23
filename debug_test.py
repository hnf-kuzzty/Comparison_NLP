#!/usr/bin/env python3
"""
Debug script to test individual components and identify issues
"""

import sys
import traceback
import numpy as np

def test_data_loading():
    """Test data loading functionality"""
    print("="*50)
    print("TESTING DATA LOADING")
    print("="*50)
    
    try:
        from data_loader import DataLoader
        
        data_loader = DataLoader()
        
        # Test tourism data loading
        print("1. Testing tourism data loading...")
        tourism_data = data_loader.load_tourism_data()
        if tourism_data is not None:
            print(f"   ✓ Tourism data loaded: {tourism_data.shape}")
        else:
            print("   ⚠ Tourism data not available (this is OK)")
        
        # Test e-commerce data creation
        print("2. Testing e-commerce data creation...")
        ecommerce_data = data_loader.load_ecommerce_data_sample()
        if ecommerce_data is not None:
            print(f"   ✓ E-commerce data created: {ecommerce_data.shape}")
            print(f"   Sample columns: {list(ecommerce_data.columns)}")
        else:
            print("   ✗ Failed to create e-commerce data")
            return False
        
        # Test data preprocessing
        print("3. Testing data preprocessing...")
        processed_data = data_loader.preprocess_data()
        if processed_data is not None and len(processed_data) > 0:
            print(f"   ✓ Processed data created: {processed_data.shape}")
            print(f"   Sample columns: {list(processed_data.columns)}")
            print(f"   Sample data:")
            print(processed_data.head(3))
        else:
            print("   ✗ Failed to preprocess data")
            return False
        
        # Test query-document pairs
        print("4. Testing query-document pairs...")
        query_doc_pairs = data_loader.get_query_document_pairs()
        if query_doc_pairs and len(query_doc_pairs) > 0:
            print(f"   ✓ Query-document pairs created: {len(query_doc_pairs)}")
            print(f"   Sample pair: {query_doc_pairs[0]['query']}")
            print(f"   Documents count: {len(query_doc_pairs[0]['documents'])}")
            print(f"   Relevance scores: {query_doc_pairs[0]['relevance_scores'][:3]}...")
        else:
            print("   ✗ Failed to create query-document pairs")
            return False
        
        return True, data_loader, processed_data, query_doc_pairs
        
    except Exception as e:
        print(f"   ✗ Error in data loading: {e}")
        traceback.print_exc()
        return False

def test_model_training(training_texts):
    """Test model training functionality"""
    print("\n" + "="*50)
    print("TESTING MODEL TRAINING")
    print("="*50)
    
    try:
        from models import TFIDFCosineSimilarity, StringMatching
        
        # Test TF-IDF model (simplest)
        print("1. Testing TF-IDF model...")
        tfidf_model = TFIDFCosineSimilarity()
        tfidf_model.train(training_texts)
        
        if tfidf_model.is_trained:
            print("   ✓ TF-IDF model trained successfully")
            
            # Test similarity calculation
            test_query = "best smartphone"
            test_docs = ["iPhone smartphone device", "luxury watch expensive", "Samsung phone mobile"]
            similarities = tfidf_model.get_similarity_scores(test_query, test_docs)
            print(f"   ✓ Similarity scores: {similarities}")
            
            return True, tfidf_model
        else:
            print("   ✗ TF-IDF model failed to train")
            return False
            
    except Exception as e:
        print(f"   ✗ Error in model training: {e}")
        traceback.print_exc()
        return False

def test_evaluation(model, query_doc_pairs):
    """Test evaluation functionality"""
    print("\n" + "="*50)
    print("TESTING EVALUATION")
    print("="*50)
    
    try:
        from evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        print("1. Testing model evaluation...")
        # Use the corrected method signature - only pass model and query_doc_pairs
        result = evaluator.evaluate_recommendation_quality(model, query_doc_pairs[:2])  # Test with 2 pairs
        
        if result:
            print("   ✓ Evaluation completed successfully")
            print(f"   Results: {result}")
            return True, evaluator
        else:
            print("   ✗ Evaluation failed")
            return False
            
    except Exception as e:
        print(f"   ✗ Error in evaluation: {e}")
        traceback.print_exc()
        return False

def main():
    print("NLP MODEL COMPARISON - DEBUG TEST")
    print("="*80)
    
    # Test 1: Data Loading
    data_result = test_data_loading()
    if not data_result:
        print("\n❌ Data loading failed. Cannot proceed.")
        return
    
    success, data_loader, processed_data, query_doc_pairs = data_result
    training_texts = processed_data['text'].tolist()
    
    # Test 2: Model Training
    model_result = test_model_training(training_texts)
    if not model_result:
        print("\n❌ Model training failed. Cannot proceed.")
        return
    
    success, model = model_result
    
    # Test 3: Evaluation
    eval_result = test_evaluation(model, query_doc_pairs)
    if not eval_result:
        print("\n❌ Evaluation failed.")
        return
    
    success, evaluator = eval_result
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("The system should work correctly now.")
    print("Run main_comparison.py to see full results.")
    print("="*80)

if __name__ == "__main__":
    main()
