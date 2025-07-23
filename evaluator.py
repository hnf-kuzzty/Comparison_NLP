import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
    
    def evaluate_recommendation_quality(self, model, query_doc_pairs):
        """Enhanced evaluation with more sophisticated metrics"""
        print(f"\nEvaluating {model.name}...")
        
        start_time = time.time()
        
        # Metrics for different relevance thresholds
        precision_at_k = {k: [] for k in [1, 3, 5, 10]}
        recall_at_k = {k: [] for k in [1, 3, 5, 10]}
        ndcg_at_k = {k: [] for k in [1, 3, 5, 10]}
        
        # Additional sophisticated metrics
        map_scores = []  # Mean Average Precision
        mrr_scores = []  # Mean Reciprocal Rank
        response_times = []
        
        for pair_data in query_doc_pairs:
            query = pair_data['query']
            documents = pair_data['documents']
            true_relevance = np.array(pair_data['relevance_scores'])
            
            if len(documents) == 0:
                continue
                
            query_start = time.time()
            
            try:
                similarities = model.get_similarity_scores(query, documents)
                if len(similarities) == 0:
                    continue
                    
                similarities = np.array(similarities)
                
                if len(similarities) != len(documents):
                    continue
                    
            except Exception as e:
                print(f"Error getting similarities for {model.name}: {e}")
                continue
            
            query_time = time.time() - query_start
            response_times.append(query_time)
            
            # Get ranking based on similarity scores
            ranking_indices = np.argsort(similarities)[::-1]
            ranked_relevance = true_relevance[ranking_indices]
            
            # Calculate Average Precision for this query
            relevant_positions = []
            cumulative_precision = []
            
            for i, relevance in enumerate(ranked_relevance):
                if relevance > 0.5:  # Consider as relevant
                    relevant_positions.append(i + 1)
                    precision_at_i = len(relevant_positions) / (i + 1)
                    cumulative_precision.append(precision_at_i)
            
            if relevant_positions:
                average_precision = np.mean(cumulative_precision)
                map_scores.append(average_precision)
                
                # Mean Reciprocal Rank - position of first relevant document
                first_relevant_pos = relevant_positions[0]
                mrr_scores.append(1.0 / first_relevant_pos)
            else:
                map_scores.append(0.0)
                mrr_scores.append(0.0)
            
            # Calculate metrics for different k values
            for k in [1, 3, 5, 10]:
                if k > len(documents):
                    k_actual = len(documents)
                else:
                    k_actual = k
                
                top_k_indices = ranking_indices[:k_actual]
                top_k_relevance = true_relevance[top_k_indices]
                
                # Precision@k with different relevance thresholds
                highly_relevant = np.sum(top_k_relevance >= 0.8)
                moderately_relevant = np.sum(top_k_relevance >= 0.5)
                
                precision_k = moderately_relevant / k_actual if k_actual > 0 else 0
                precision_at_k[k].append(precision_k)
                
                # Recall@k
                total_relevant = np.sum(true_relevance >= 0.5)
                if total_relevant > 0:
                    recall_k = moderately_relevant / total_relevant
                else:
                    recall_k = 0
                recall_at_k[k].append(recall_k)
                
                # NDCG@k with graded relevance
                if k_actual > 0:
                    dcg = np.sum(top_k_relevance / np.log2(np.arange(2, k_actual + 2)))
                    ideal_relevance = np.sort(true_relevance)[::-1][:k_actual]
                    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k_actual + 2)))
                    ndcg = dcg / idcg if idcg > 0 else 0
                else:
                    ndcg = 0
                ndcg_at_k[k].append(ndcg)
        
        total_time = time.time() - start_time
        
        # Calculate average metrics
        avg_precision_5 = np.mean(precision_at_k[5]) if precision_at_k[5] else 0
        avg_recall_5 = np.mean(recall_at_k[5]) if recall_at_k[5] else 0
        avg_ndcg_5 = np.mean(ndcg_at_k[5]) if ndcg_at_k[5] else 0
        avg_map = np.mean(map_scores) if map_scores else 0
        avg_mrr = np.mean(mrr_scores) if mrr_scores else 0
        avg_response_time = np.mean(response_times) if response_times else 0
        
        f1_score = 2 * (avg_precision_5 * avg_recall_5) / (avg_precision_5 + avg_recall_5) if (avg_precision_5 + avg_recall_5) > 0 else 0
        
        # Store comprehensive results
        self.results[model.name] = {
            'precision': avg_precision_5,
            'recall': avg_recall_5,
            'f1_score': f1_score,
            'ndcg': avg_ndcg_5,
            'map': avg_map,
            'mrr': avg_mrr,
            'avg_response_time': avg_response_time,
            'total_time': total_time,
            'queries_processed': len(query_doc_pairs),
            'precision_at_k': {k: np.mean(v) if v else 0 for k, v in precision_at_k.items()},
            'recall_at_k': {k: np.mean(v) if v else 0 for k, v in recall_at_k.items()},
            'ndcg_at_k': {k: np.mean(v) if v else 0 for k, v in ndcg_at_k.items()}
        }
        
        print(f"Completed evaluation for {model.name}")
        print(f"Precision@5: {avg_precision_5:.4f}")
        print(f"Recall@5: {avg_recall_5:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"NDCG@5: {avg_ndcg_5:.4f}")
        print(f"MAP: {avg_map:.4f}")
        print(f"MRR: {avg_mrr:.4f}")
        print(f"Avg Response Time: {avg_response_time:.4f}s")
        
        return self.results[model.name]
    
    def compare_models(self):
        """Compare all evaluated models with comprehensive metrics"""
        if not self.results:
            print("No models have been evaluated yet.")
            return None
        
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("="*100)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Precision@5': metrics['precision'],
                'Recall@5': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'NDCG@5': metrics['ndcg'],
                'MAP': metrics['map'],
                'MRR': metrics['mrr'],
                'Response Time (s)': metrics['avg_response_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        print("\nPerformance Ranking (by F1-Score):")
        print("-" * 100)
        for i, (idx, row) in enumerate(df.iterrows(), 1):
            print(f"{i:2d}. {row['Model']:<35} F1: {row['F1-Score']:.4f} | MAP: {row['MAP']:.4f} | NDCG@5: {row['NDCG@5']:.4f}")
        
        print(f"\nDetailed Metrics:")
        print("-" * 100)
        print(df.to_string(index=False, float_format='%.4f'))
        
        return df
    
    def plot_comparison(self):
        """Create comprehensive visualization of model comparison"""
        if not self.results:
            print("No results to plot.")
            return
        
        try:
            models = list(self.results.keys())
            
            # Create a more comprehensive comparison
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # F1-Score comparison (most important)
            f1_scores = [self.results[model]['f1_score'] for model in models]
            bars1 = ax1.bar(range(len(models)), f1_scores, color='lightgreen')
            ax1.set_title('F1-Score Comparison (Primary Metric)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('F1-Score')
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels([m[:12] + '...' if len(m) > 12 else m for m in models], rotation=45, ha='right')
            ax1.set_ylim(0, max(max(f1_scores), 0.1) * 1.1)
            
            for bar, score in zip(bars1, f1_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + max(f1_scores) * 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # MAP comparison
            map_scores = [self.results[model]['map'] for model in models]
            bars2 = ax2.bar(range(len(models)), map_scores, color='skyblue')
            ax2.set_title('Mean Average Precision (MAP)', fontsize=14)
            ax2.set_ylabel('MAP')
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels([m[:12] + '...' if len(m) > 12 else m for m in models], rotation=45, ha='right')
            ax2.set_ylim(0, max(max(map_scores), 0.1) * 1.1)
            
            for bar, score in zip(bars2, map_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + max(map_scores) * 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            # NDCG@5 comparison
            ndcg_scores = [self.results[model]['ndcg'] for model in models]
            bars3 = ax3.bar(range(len(models)), ndcg_scores, color='lightcoral')
            ax3.set_title('NDCG@5 (Ranking Quality)', fontsize=14)
            ax3.set_ylabel('NDCG@5')
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels([m[:12] + '...' if len(m) > 12 else m for m in models], rotation=45, ha='right')
            ax3.set_ylim(0, max(max(ndcg_scores), 0.1) * 1.1)
            
            for bar, score in zip(bars3, ndcg_scores):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + max(ndcg_scores) * 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Response time comparison (log scale for better visibility)
            response_times = [self.results[model]['avg_response_time'] for model in models]
            bars4 = ax4.bar(range(len(models)), response_times, color='gold')
            ax4.set_title('Average Response Time', fontsize=14)
            ax4.set_ylabel('Response Time (seconds)')
            ax4.set_yscale('log')
            ax4.set_xticks(range(len(models)))
            ax4.set_xticklabels([m[:12] + '...' if len(m) > 12 else m for m in models], rotation=45, ha='right')
            
            for bar, time in zip(bars4, response_times):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2, height * 1.1,
                        f'{time:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Comprehensive comparison plot saved as 'comprehensive_model_comparison.png'")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        if not self.results:
            print("No results available for report generation.")
            return ""
        
        try:
            report = []
            report.append("COMPREHENSIVE NLP MODEL COMPARISON REPORT")
            report.append("=" * 60)
            report.append("")
            
            # Executive Summary
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 30)
            
            best_f1_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
            best_map_model = max(self.results.keys(), key=lambda x: self.results[x]['map'])
            fastest_model = min(self.results.keys(), key=lambda x: self.results[x]['avg_response_time'])
            
            report.append(f"Best Overall Performance (F1-Score): {best_f1_model}")
            report.append(f"  F1-Score: {self.results[best_f1_model]['f1_score']:.4f}")
            report.append(f"  MAP: {self.results[best_f1_model]['map']:.4f}")
            report.append(f"  NDCG@5: {self.results[best_f1_model]['ndcg']:.4f}")
            report.append("")
            
            report.append(f"Best Ranking Quality (MAP): {best_map_model}")
            report.append(f"  MAP: {self.results[best_map_model]['map']:.4f}")
            report.append("")
            
            report.append(f"Fastest Model: {fastest_model}")
            report.append(f"  Avg Response Time: {self.results[fastest_model]['avg_response_time']:.4f}s")
            report.append("")
            
            # Performance Analysis
            report.append("PERFORMANCE ANALYSIS")
            report.append("-" * 30)
            
            f1_scores = [self.results[model]['f1_score'] for model in self.results.keys()]
            f1_std = np.std(f1_scores)
            
            if f1_std < 0.05:
                report.append("⚠️  LOW DISCRIMINATION: Models show very similar performance")
                report.append("   Consider more challenging evaluation scenarios")
            elif f1_std > 0.2:
                report.append("✅ HIGH DISCRIMINATION: Clear performance differences between models")
            else:
                report.append("✅ MODERATE DISCRIMINATION: Reasonable performance differences")
            
            report.append(f"   F1-Score Standard Deviation: {f1_std:.4f}")
            report.append("")
            
            # Detailed Results
            report.append("DETAILED RESULTS")
            report.append("-" * 30)
            
            for model_name, metrics in sorted(self.results.items(), 
                                            key=lambda x: x[1]['f1_score'], reverse=True):
                report.append(f"\n{model_name}:")
                report.append(f"  Precision@5: {metrics['precision']:.4f}")
                report.append(f"  Recall@5: {metrics['recall']:.4f}")
                report.append(f"  F1-Score: {metrics['f1_score']:.4f}")
                report.append(f"  NDCG@5: {metrics['ndcg']:.4f}")
                report.append(f"  MAP: {metrics['map']:.4f}")
                report.append(f"  MRR: {metrics['mrr']:.4f}")
                report.append(f"  Avg Response Time: {metrics['avg_response_time']:.4f}s")
            
            report_text = "\n".join(report)
            
            # Save report to file
            with open('comprehensive_evaluation_report.txt', 'w') as f:
                f.write(report_text)
            
            print(report_text)
            print(f"\nComprehensive report saved to 'comprehensive_evaluation_report.txt'")
            
            return report_text
            
        except Exception as e:
            print(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return ""
