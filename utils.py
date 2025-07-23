import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

class AnalysisUtils:
    @staticmethod
    def generate_test_queries():
        """Generate test queries for evaluation"""
        queries = [
            # E-commerce related
            "best smartphone under 500",
            "comfortable running shoes",
            "laptop for gaming",
            "wireless headphones review",
            "kitchen appliances sale",
            "summer dress collection",
            "fitness equipment home",
            "organic skincare products",
            
            # Tourism related
            "beach vacation packages",
            "mountain hiking tours",
            "city break weekend",
            "luxury hotel booking",
            "budget travel tips",
            "family friendly resorts",
            "adventure travel destinations",
            "cultural heritage sites",
            
            # General search queries
            "how to cook pasta",
            "weather forecast today",
            "movie recommendations",
            "book reviews fiction",
            "health and wellness",
            "technology news updates",
            "investment advice",
            "home improvement ideas"
        ]
        return queries
    
    @staticmethod
    def create_query_pairs_for_similarity():
        """Create query pairs to test semantic similarity"""
        pairs = [
            ("best smartphone", "top mobile phone"),
            ("cheap laptop", "affordable computer"),
            ("vacation packages", "holiday deals"),
            ("running shoes", "athletic footwear"),
            ("healthy recipes", "nutritious cooking"),
            ("movie reviews", "film ratings"),
            ("travel tips", "journey advice"),
            ("fitness equipment", "exercise gear"),
            ("book recommendations", "reading suggestions"),
            ("weather forecast", "climate prediction")
        ]
        return pairs
    
    @staticmethod
    def analyze_dataset_characteristics(data):
        """Analyze characteristics of the dataset"""
        print("DATASET ANALYSIS")
        print("=" * 50)
        
        if isinstance(data, pd.DataFrame):
            print(f"Dataset shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print("\nData types:")
            print(data.dtypes)
            
            print("\nMissing values:")
            print(data.isnull().sum())
            
            # Analyze text columns
            text_columns = data.select_dtypes(include=['object']).columns
            for col in text_columns:
                if data[col].dtype == 'object':
                    print(f"\n{col} analysis:")
                    print(f"  Unique values: {data[col].nunique()}")
                    print(f"  Most common values:")
                    print(f"  {data[col].value_counts().head()}")
        
        return data.describe()
    
    @staticmethod
    def create_word_cloud(texts, title="Word Cloud"):
        """Create word cloud from text data"""
        try:
            # Combine all texts
            combined_text = " ".join([str(text) for text in texts if pd.notna(text)])
            
            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=100).generate(combined_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating word cloud: {e}")
    
    @staticmethod
    def plot_text_length_distribution(texts, title="Text Length Distribution"):
        """Plot distribution of text lengths"""
        lengths = [len(str(text).split()) for text in texts if pd.notna(text)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_length = np.mean(lengths)
        median_length = np.median(lengths)
        plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.1f}')
        plt.axvline(median_length, color='green', linestyle='--', label=f'Median: {median_length:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Text length statistics:")
        print(f"  Mean: {mean_length:.2f} words")
        print(f"  Median: {median_length:.2f} words")
        print(f"  Min: {min(lengths)} words")
        print(f"  Max: {max(lengths)} words")
    
    @staticmethod
    def extract_keywords(texts, top_n=20):
        """Extract most common keywords from texts"""
        # Combine all texts and clean
        combined_text = " ".join([str(text).lower() for text in texts if pd.notna(text)])
        
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        top_keywords = word_counts.most_common(top_n)
        
        print(f"Top {top_n} Keywords:")
        print("-" * 30)
        for i, (word, count) in enumerate(top_keywords, 1):
            print(f"{i:2d}. {word:<15} ({count} occurrences)")
        
        return top_keywords
    
    @staticmethod
    def save_results_to_csv(results, filename="model_comparison_results.csv"):
        """Save comparison results to CSV"""
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(filename)
        print(f"Results saved to {filename}")
        return df
