import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self):
        self.tourism_data = None
        self.ecommerce_data = None
        self.processed_data = None
        self.steam_data = None
        
    def load_tourism_data(self, file_path="data/customer_behaviour_tourism.csv"):
        """Load the tourism dataset"""
        try:
            self.tourism_data = pd.read_csv(file_path)
            print(f"Tourism data loaded: {self.tourism_data.shape}")
            return self.tourism_data
        except Exception as e:
            print(f"Error loading tourism data: {e}")
            return None
    
    def load_ecommerce_data_sample(self):
        """Create a sample e-commerce dataset"""
        np.random.seed(42)
        n_samples = 500
        
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
        products = [
            'smartphone', 'laptop', 'headphones', 'tablet', 'camera',
            'dress', 'jeans', 'shoes', 'jacket', 'shirt',
            'novel', 'cookbook', 'textbook', 'magazine', 'comic'
        ]
        
        data = {
            'user_id': np.random.randint(1, 201, n_samples),
            'product_name': np.random.choice(products, n_samples),
            'category': np.random.choice(categories, n_samples),
            'search_query': [],
            'rating': np.random.uniform(1, 5, n_samples),
            'review_text': [],
            'purchase_intent': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        }
        
        for i in range(n_samples):
            product = data['product_name'][i]
            category = data['category'][i]
            
            query_templates = [
                f"best {product}",
                f"cheap {product}",
                f"{product} reviews",
                f"buy {product} online",
                f"{category} {product}"
            ]
            data['search_query'].append(np.random.choice(query_templates))
            
            if data['rating'][i] >= 4:
                review_templates = [
                    f"Great {product}, highly recommend!",
                    f"Excellent quality {product}, very satisfied",
                    f"Perfect {product} for the price"
                ]
            else:
                review_templates = [
                    f"Poor quality {product}, disappointed",
                    f"Not worth the money, {product} broke quickly",
                    f"Bad {product}, would not recommend"
                ]
            data['review_text'].append(np.random.choice(review_templates))
        
        self.ecommerce_data = pd.DataFrame(data)
        print(f"Sample e-commerce data created: {self.ecommerce_data.shape}")
        return self.ecommerce_data
    
    def load_steam_games_data(self):
        """Load Steam games data from local files ONLY - no Kaggle download"""
        print("📁 Looking for local Steam games data...")
        
        # Define file paths
        games_path = 'data/games.csv'
        recommendations_path = 'data/recommendations.csv'
        users_path = 'data/users.csv'
        
        # Check if games.csv exists
        if not os.path.exists(games_path):
            print(f"❌ {games_path} not found!")
            print("Creating sample gaming data instead...")
            return self.create_sample_gaming_data()
        
        # Load games.csv
        try:
            print(f"✅ Loading {games_path}...")
            games_df = pd.read_csv(games_path)
            print(f"   Shape: {games_df.shape}")
            print(f"   Columns: {list(games_df.columns)[:5]}...")
            
            # Start with games as our base dataset
            self.steam_data = games_df
            
            # Try to load recommendations if available
            if os.path.exists(recommendations_path):
                print(f"✅ Loading {recommendations_path}...")
                try:
                    rec_df = pd.read_csv(recommendations_path)
                    print(f"   Shape: {rec_df.shape}")
                    print(f"   Columns: {list(rec_df.columns)}")
                    
                    # Find common columns for merging
                    common_cols = set(games_df.columns) & set(rec_df.columns)
                    if common_cols:
                        merge_col = list(common_cols)[0]
                        print(f"   Merging on column: {merge_col}")
                        
                        # Aggregate recommendations
                        rec_agg = rec_df.groupby(merge_col).agg({
                            col: 'first' for col in rec_df.columns if col != merge_col
                        }).reset_index()
                        
                        # Merge with games
                        self.steam_data = games_df.merge(rec_agg, on=merge_col, how='left', suffixes=('', '_rec'))
                        print(f"   Merged shape: {self.steam_data.shape}")
                    else:
                        print("   ⚠️ No common columns found for merging recommendations")
                except Exception as e:
                    print(f"   ⚠️ Error processing recommendations: {e}")
            
            # Try to load users if available
            if os.path.exists(users_path):
                print(f"✅ Found {users_path} (not merging, but available for future use)")
            
            print(f"✅ Steam data ready: {self.steam_data.shape}")
            return self.steam_data
            
        except Exception as e:
            print(f"❌ Error loading games data: {e}")
            print("Creating sample gaming data instead...")
            return self.create_sample_gaming_data()
    
    def create_sample_gaming_data(self):
        """Create sample gaming data if real data not available"""
        print("🎮 Creating sample gaming data...")
        
        np.random.seed(42)
        n_games = 300
        
        genres = ['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Racing', 'Puzzle', 'Indie']
        developers = ['Studio A', 'Studio B', 'Studio C', 'Indie Dev', 'Big Corp', 'Small Team']
        
        game_names = [
            'Epic Quest', 'Space Battle', 'City Builder', 'Racing Pro', 'Puzzle Master',
            'Adventure Land', 'Strategy War', 'Indie Game', 'Sports Champion', 'Simulation Life'
        ]
        
        descriptions = [
            'Epic fantasy adventure with magic and dragons',
            'Fast-paced action shooter with multiplayer modes',
            'Strategic city building and resource management',
            'Immersive role-playing game with character progression',
            'Realistic racing simulation with licensed cars',
            'Challenging puzzle game with unique mechanics',
            'Indie platformer with beautiful pixel art',
            'Competitive multiplayer battle arena',
            'Relaxing simulation game for casual players'
        ]
        
        data = {
            'app_id': range(1000, 1000 + n_games),
            'title': [f"{np.random.choice(game_names)} {i}" for i in range(n_games)],
            'genre': np.random.choice(genres, n_games),
            'description': np.random.choice(descriptions, n_games),
            'developer': np.random.choice(developers, n_games),
            'positive_reviews': np.random.randint(0, 5000, n_games),
            'negative_reviews': np.random.randint(0, 500, n_games),
            'price': np.random.uniform(0, 60, n_games),
            'release_date': pd.date_range('2015-01-01', '2024-01-01', periods=n_games)
        }
        
        self.steam_data = pd.DataFrame(data)
        print(f"✅ Sample gaming data created: {self.steam_data.shape}")
        return self.steam_data
    
    def preprocess_data(self):
        """Create processed data for training"""
        print("🔄 Preprocessing data...")
        
        processed_samples = []
        
        # Load all data sources
        steam_data = self.load_steam_games_data()
        ecommerce_data = self.load_ecommerce_data_sample()
        tourism_data = self.load_tourism_data()
        
        # Gaming scenarios
        gaming_scenarios = [
            {
                'query': 'best RPG games with character customization',
                'docs': [
                    'Epic fantasy RPG with deep character creation and skill trees',
                    'Immersive role-playing game featuring extensive character customization',
                    'Open-world RPG with detailed character progression system'
                ],
                'relevance': [1.0, 0.9, 0.8]
            },
            {
                'query': 'multiplayer competitive games for esports',
                'docs': [
                    'Professional esports title with ranked competitive matchmaking',
                    'Team-based multiplayer shooter designed for tournaments',
                    'Strategic multiplayer game with established esports scene'
                ],
                'relevance': [1.0, 0.9, 0.8]
            },
            {
                'query': 'indie games with unique art style',
                'docs': [
                    'Beautiful indie platformer with hand-drawn pixel art',
                    'Artistic indie game with unique visual storytelling',
                    'Creative indie title with innovative art direction'
                ],
                'relevance': [1.0, 0.9, 0.8]
            }
        ]
        
        # Add gaming scenarios
        for scenario in gaming_scenarios:
            query = scenario['query']
            for doc, rel in zip(scenario['docs'], scenario['relevance']):
                processed_samples.append({
                    'text': doc,
                    'query': query,
                    'source': 'synthetic_gaming',
                    'label': 'relevant' if rel > 0.5 else 'irrelevant',
                    'relevance_score': rel
                })
        
        # Process Steam games data
        if steam_data is not None:
            print(f"  Processing {len(steam_data)} Steam games...")
            
            gaming_queries = [
                'best indie games',
                'multiplayer action games', 
                'RPG games with good story',
                'strategy games for beginners',
                'racing games realistic'
            ]
            
            for _, game in steam_data.iterrows():
                # Create game text
                text_parts = []
                
                if 'title' in game and pd.notna(game['title']):
                    text_parts.append(str(game['title']))
                
                if 'genre' in game and pd.notna(game['genre']):
                    text_parts.append(f"Genre: {game['genre']}")
                
                if 'description' in game and pd.notna(game['description']):
                    text_parts.append(str(game['description']))
                
                if 'developer' in game and pd.notna(game['developer']):
                    text_parts.append(f"Developer: {game['developer']}")
                
                if text_parts:
                    game_text = ' '.join(text_parts)
                    
                    # Assign to relevant queries
                    assigned_query = np.random.choice(gaming_queries)
                    
                    # Calculate relevance based on reviews
                    base_relevance = 0.6
                    if 'positive_reviews' in game and 'negative_reviews' in game:
                        pos = game['positive_reviews'] if pd.notna(game['positive_reviews']) else 0
                        neg = game['negative_reviews'] if pd.notna(game['negative_reviews']) else 0
                        total = pos + neg
                        if total > 0:
                            base_relevance = 0.3 + (pos / total) * 0.7
                    
                    # Add noise
                    relevance = max(0, min(1, base_relevance + np.random.normal(0, 0.1)))
                    
                    processed_samples.append({
                        'text': game_text,
                        'query': assigned_query,
                        'source': 'steam_games',
                        'label': 'relevant' if relevance > 0.5 else 'irrelevant',
                        'relevance_score': relevance
                    })
        
        # Process e-commerce data
        if ecommerce_data is not None:
            print(f"  Processing {len(ecommerce_data)} e-commerce items...")
            
            for _, item in ecommerce_data.iterrows():
                query = item['search_query']
                text = f"{item['product_name']} {item['category']} {item['review_text']}"
                relevance = 0.6 if item['rating'] >= 4 else 0.4
                relevance += np.random.normal(0, 0.1)
                relevance = max(0, min(1, relevance))
                
                processed_samples.append({
                    'text': text,
                    'query': query,
                    'source': 'ecommerce',
                    'label': 'relevant' if relevance > 0.5 else 'irrelevant',
                    'relevance_score': relevance
                })
        
        # Process tourism data
        if tourism_data is not None:
            print(f"  Processing {len(tourism_data)} tourism records...")
            
            for _, row in tourism_data.iterrows():
                text_parts = []
                for col in tourism_data.columns:
                    if tourism_data[col].dtype == 'object' and pd.notna(row[col]):
                        text_parts.append(str(row[col]))
                
                if text_parts:
                    processed_samples.append({
                        'text': ' '.join(text_parts),
                        'query': 'tourism behavior analysis',
                        'source': 'tourism',
                        'label': 'relevant',
                        'relevance_score': 0.8
                    })
        
        self.processed_data = pd.DataFrame(processed_samples)
        print(f"✅ Processed data created: {self.processed_data.shape}")
        print(f"   Sources: {dict(self.processed_data['source'].value_counts())}")
        
        return self.processed_data
    
    def get_query_document_pairs(self):
        """Get query-document pairs for evaluation"""
        if self.processed_data is None:
            self.preprocess_data()
        
        query_doc_pairs = []
        queries = self.processed_data['query'].unique()
        
        for query in queries:
            query_data = self.processed_data[self.processed_data['query'] == query]
            documents = query_data['text'].tolist()
            relevance_scores = query_data['relevance_score'].tolist()
            
            # Shuffle to avoid bias
            combined = list(zip(documents, relevance_scores))
            np.random.shuffle(combined)
            documents, relevance_scores = zip(*combined)
            
            query_doc_pairs.append({
                'query': query,
                'documents': list(documents),
                'relevance_scores': list(relevance_scores)
            })
        
        return query_doc_pairs
    
    def get_train_test_split(self, test_size=0.2):
        """Split data for training and testing"""
        if self.processed_data is None:
            self.preprocess_data()
        
        X = self.processed_data['text'].values
        y = self.processed_data['label'].values
        
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
