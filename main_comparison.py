#!/usr/bin/env python3
"""
NLP Model Comparison for Recommendation Systems
Main comparison script that evaluates all models and generates comprehensive results.
"""

import sys
import warnings
import traceback
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import DataLoader
from models import (
    TFIDFCosineSimilarity, StringMatching, BERTSemanticModel,
    Word2VecCosineSimilarity, RDFEmbeddingModel, WordMoversDistance,
    SentimentFeatureModel
)
from evaluator import ModelEvaluator
from utils import AnalysisUtils

def main():
    print("="*80)
    print("NLP MODEL COMPARISON FOR RECOMMENDATION SYSTEMS")
    print("="*80)
    print("Comparing 7 different NLP models for user search query recommendations")
    print()
    
    try:
        # Step 1: Load and preprocess data
        print("Step 1: Loading and preprocessing data...")
        print("-" * 50)
        
        data_loader = DataLoader()
        
        # Load tourism data
        print("Loading tourism data...")
        tourism_data = data_loader.load_tourism_data()
        
        # Create sample e-commerce data
        print("Creating e-commerce sample data...")
        ecommerce_data = data_loader.load_ecommerce_data_sample()
        
        # Preprocess and combine data
        print("Preprocessing data...")
        processed_data = data_loader.preprocess_data()
        
        if processed_data is None or len(processed_data) == 0:
            print("Error: No data available for training. Exiting.")
            return
        
        print(f"✓ Successfully processed {len(processed_data)} data samples")
        
        # Get query-document pairs for evaluation
        query_doc_pairs = data_loader.get_query_document_pairs()
        print(f"✓ Created {len(query_doc_pairs)} query-document pairs for evaluation")
        
        # Extract training texts
        training_texts = processed_data['text'].tolist()
        training_labels = processed_data['label'].tolist()
        
        print(f"✓ Total training samples: {len(training_texts)}")
        
        # Step 2: Initialize all models
        print("\nStep 2: Initializing models...")
        print("-" * 50)
        
        models = [
            TFIDFCosineSimilarity(),           # Baseline 1
            StringMatching(),                  # Baseline 2
            BERTSemanticModel(),              # Model 1
            Word2VecCosineSimilarity(),       # Model 2
            RDFEmbeddingModel(),              # Model 3
            WordMoversDistance(),             # Model 4
            SentimentFeatureModel()           # Model 5
        ]
        
        print(f"✓ Initialized {len(models)} models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model.name}")
        
        # Step 3: Train all models
        print("\nStep 3: Training models...")
        print("-" * 50)
        
        successfully_trained = []
        
        for model in models:
            try:
                print(f"\nTraining {model.name}...")
                model.train(training_texts, training_labels)
                if model.is_trained:
                    successfully_trained.append(model)
                    print(f"✓ {model.name} trained successfully")
                else:
                    print(f"✗ {model.name} failed to train")
            except Exception as e:
                print(f"✗ Error training {model.name}: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n✓ Successfully trained {len(successfully_trained)} out of {len(models)} models")
        
        if len(successfully_trained) == 0:
            print("❌ No models were successfully trained. Cannot proceed with evaluation.")
            return
        
        # Step 4: Test model functionality
        print("\nStep 4: Testing model functionality...")
        print("-" * 50)
        
        test_query = "best smartphone under 500"
        test_docs = ["iPhone 12 smartphone", "expensive luxury watch", "Samsung Galaxy phone"]
        
        working_models = []
        for model in successfully_trained:
            try:
                print(f"Testing {model.name}...")
                similarities = model.get_similarity_scores(test_query, test_docs)
                if len(similarities) > 0:
                    working_models.append(model)
                    print(f"✓ {model.name} working - similarity scores: {similarities}")
                else:
                    print(f"✗ {model.name} returned empty similarities")
            except Exception as e:
                print(f"✗ Error testing {model.name}: {e}")
                traceback.print_exc()
        
        print(f"\n✓ {len(working_models)} models are working correctly")
        
        if len(working_models) == 0:
            print("❌ No models are working correctly. Cannot proceed with evaluation.")
            return
        
        # Step 5: Evaluate all working models
        print("\nStep 5: Evaluating models...")
        print("-" * 50)
        
        evaluator = ModelEvaluator()
        evaluated_models = 0
        
        for model in working_models:
            try:
                print(f"\nEvaluating {model.name}...")
                # Evaluate recommendation quality
                result = evaluator.evaluate_recommendation_quality(model, query_doc_pairs)
                if result:
                    evaluated_models += 1
                    print(f"✓ {model.name} evaluated successfully")
                else:
                    print(f"✗ {model.name} evaluation failed")
                
            except Exception as e:
                print(f"✗ Error evaluating {model.name}: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n✓ Successfully evaluated {evaluated_models} models")
        
        if evaluated_models == 0:
            print("❌ No models were successfully evaluated.")
            return
        
        # Step 6: Compare results and generate report
        print("\nStep 6: Generating comparison results...")
        print("-" * 50)
        
        # Compare all models
        comparison_df = evaluator.compare_models()
        
        # Create visualizations
        print("\nGenerating visualizations...")
        try:
            evaluator.plot_comparison()
            print("✓ Comparison plots generated")
        except Exception as e:
            print(f"✗ Error creating plots: {e}")
            traceback.print_exc()
        
        # Generate comprehensive report
        print("\nGenerating comprehensive report...")
        try:
            evaluator.generate_report()
            print("✓ Report generated")
        except Exception as e:
            print(f"✗ Error generating report: {e}")
            traceback.print_exc()
        
        # Save results to CSV
        try:
            AnalysisUtils.save_results_to_csv(evaluator.results)
            print("✓ Results saved to CSV")
        except Exception as e:
            print(f"✗ Error saving results: {e}")
        
        # Step 7: Summary and recommendations
        print("\n" + "="*80)
        print("EVALUATION COMPLETE - SUMMARY")
        print("="*80)
        
        if evaluator.results:
            best_model = max(evaluator.results.keys(), key=lambda x: evaluator.results[x]['f1_score'])
            best_f1 = evaluator.results[best_model]['f1_score']
            
            print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
            print(f"   F1-Score: {best_f1:.4f}")
            print(f"   Precision: {evaluator.results[best_model]['precision']:.4f}")
            print(f"   Recall: {evaluator.results[best_model]['recall']:.4f}")
            
            fastest_model = min(evaluator.results.keys(), key=lambda x: evaluator.results[x]['avg_response_time'])
            fastest_time = evaluator.results[fastest_model]['avg_response_time']
            
            print(f"\n⚡ FASTEST MODEL: {fastest_model}")
            print(f"   Avg Response Time: {fastest_time:.4f}s")
            
            print(f"\n📊 TOTAL MODELS EVALUATED: {len(evaluator.results)}")
            print(f"📝 TOTAL QUERY-DOC PAIRS: {len(query_doc_pairs)}")
            print(f"📄 TOTAL DOCUMENTS PROCESSED: {len(training_texts)}")
            
            print(f"\n🎯 RECOMMENDATIONS:")
            if best_f1 > 0.7:
                print(f"   • {best_model} is ready for production deployment")
            elif best_f1 > 0.5:
                print(f"   • {best_model} shows promise but needs optimization")
            else:
                print(f"   • All models need significant improvement")
        
        print(f"\n✅ Comparison complete! Check the generated files for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Unexpected error in main execution: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nCritical error: {e}")
        traceback.print_exc()
        sys.exit(1)
