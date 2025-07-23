import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO
import kagglehub

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
        """Create a sample e-commerce dataset since we can't directly access Kaggle"""
        np.random.seed(42)
        n_samples = 1000
        
        # Sample product categories and descriptions
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
        products = [
            'smartphone', 'laptop', 'headphones', 'tablet', 'camera',
            'dress', 'jeans', 'shoes', 'jacket', 'shirt',
            'novel', 'cookbook', 'textbook', 'magazine', 'comic',
            'furniture', 'kitchen appliance', 'garden tool', 'decoration', 'lighting',
            'fitness equipment', 'sports gear', 'outdoor equipment', 'ball', 'bike',
            'skincare', 'makeup', 'perfume', 'hair care', 'wellness'
        ]
        
        # Generate sample data
        data = {
            'user_id': np.random.randint(1, 201, n_samples),
            'product_name': np.random.choice(products, n_samples),
            'category': np.random.choice(categories, n_samples),
            'search_query': [],
            'rating': np.random.uniform(1, 5, n_samples),
            'review_text': [],
            'purchase_intent': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        }
        
        # Generate search queries and reviews
        for i in range(n_samples):
            product = data['product_name'][i]
            category = data['category'][i]
            
            # Generate search query
            query_templates = [
                f"best {product}",
                f"cheap {product}",
                f"{product} reviews",
                f"buy {product} online",
                f"{category} {product}",
                f"top rated {product}"
            ]
            data['search_query'].append(np.random.choice(query_templates))
            
            # Generate review text
            if data['rating'][i] >= 4:
                review_templates = [
                    f"Great {product}, highly recommend!",
                    f"Excellent quality {product}, very satisfied",
                    f"Perfect {product} for the price",
                    f"Amazing {product}, will buy again"
                ]
            else:
                review_templates = [
                    f"Poor quality {product}, disappointed",
                    f"Not worth the money, {product} broke quickly",
                    f"Bad {product}, would not recommend",
                    f"Terrible {product}, waste of money"
                ]
            data['review_text'].append(np.random.choice(review_templates))
        
        self.ecommerce_data = pd.DataFrame(data)
        print(f"Sample e-commerce data created: {self.ecommerce_data.shape}")
        return self.ecommerce_data
    
    def load_steam_games_data(self):
        """Load Steam game recommendations dataset from Kaggle"""
        try:
            print("📥 Downloading Steam games dataset from Kaggle...")
            
            # Download latest version
            path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")
            print(f"Dataset downloaded to: {path}")
            
            # Try to find and load the main dataset file
            import os
            dataset_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        dataset_files.append(os.path.join(root, file))
        
            print(f"Found CSV files: {[os.path.basename(f) for f in dataset_files]}")
        
            # Load the main dataset (usually the largest CSV file)
            if dataset_files:
                # Sort by file size and take the largest
                dataset_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
                main_file = dataset_files[0]
            
                print(f"Loading main dataset: {os.path.basename(main_file)}")
                self.steam_data = pd.read_csv(main_file)
            
                print(f"Steam games data loaded: {self.steam_data.shape}")
                print(f"Columns: {list(self.steam_data.columns)}")
            
                # Show sample data
                print("\nSample data:")
                print(self.steam_data.head(3))
            
                return self.steam_data
            else:
                print("❌ No CSV files found in the dataset")
                return None
            
        except Exception as e:
            print(f"❌ Error loading Steam games data: {e}")
            print("Creating fallback gaming data...")
            return self.create_fallback_gaming_data()

    def create_fallback_gaming_data(self):
        """Create fallback gaming data if Kaggle download fails"""
        np.random.seed(42)
        n_samples = 500
    
        # Gaming categories and genres
        genres = ['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Racing', 'Puzzle', 'Indie', 'Casual']
        platforms = ['PC', 'Mac', 'Linux', 'Steam Deck']
    
        # Sample game names and descriptions
        game_templates = [
            "Epic fantasy adventure with magic and dragons",
            "Fast-paced action shooter with multiplayer modes",
            "Strategic city building and resource management",
            "Immersive role-playing game with character progression",
            "Realistic racing simulation with licensed cars",
            "Challenging puzzle game with unique mechanics",
            "Indie platformer with beautiful pixel art",
            "Competitive multiplayer battle arena",
            "Relaxing simulation game for casual players",
            "Story-driven adventure with meaningful choices"
        ]
    
        # Generate sample gaming data
        data = {
            'app_id': np.random.randint(1000, 999999, n_samples),
            'title': [f"Game {i}" for i in range(n_samples)],
            'genre': np.random.choice(genres, n_samples),
            'description': np.random.choice(game_templates, n_samples),
            'positive_reviews': np.random.randint(0, 10000, n_samples),
            'negative_reviews': np.random.randint(0, 1000, n_samples),
            'price': np.random.uniform(0, 60, n_samples),
            'platforms': np.random.choice(platforms, n_samples),
            'release_date': pd.date_range('2010-01-01', '2024-01-01', periods=n_samples),
            'developer': [f"Studio {chr(65 + i % 26)}" for i in range(n_samples)],
            'tags': [f"tag{i%10}, tag{(i+1)%10}, tag{(i+2)%10}" for i in range(n_samples)]
        }
    
        self.steam_data = pd.DataFrame(data)
        print(f"Fallback gaming data created: {self.steam_data.shape}")
        return self.steam_data
    
    def preprocess_data(self):
        """Create challenging and realistic query-document scenarios including Steam games"""
        processed_samples = []
    
        # Load Steam games data
        print("Loading Steam games dataset...")
        steam_data = self.load_steam_games_data()
    
        # Create more challenging search scenarios with nuanced relevance
        challenging_scenarios = [
            # Gaming-specific scenarios
            {
                'query': 'best RPG games with character customization',
                'highly_relevant': [
                    'Epic fantasy RPG with deep character creation and skill trees',
                    'Immersive role-playing game featuring extensive character customization options',
                    'Open-world RPG with detailed character progression and appearance editor'
                ],
                'moderately_relevant': [
                    'Action RPG with some character upgrade mechanics',
                    'Story-driven game with basic character choices',
                    'Adventure game with light RPG elements and customization'
                ],
                'slightly_relevant': [
                    'Action game with minimal character progression',
                    'Strategy game with unit customization features',
                    'Simulation game with avatar creation tools'
                ],
                'irrelevant': [
                    'Pure puzzle game with no character elements',
                    'Racing simulator with licensed vehicles',
                    'Music rhythm game with preset characters'
                ]
            },
        
            {
                'query': 'multiplayer competitive games for esports',
                'highly_relevant': [
                    'Professional esports title with ranked competitive matchmaking',
                    'Team-based multiplayer shooter designed for competitive tournaments',
                    'Strategic multiplayer game with established esports scene'
                ],
                'moderately_relevant': [
                    'Online multiplayer game with competitive modes',
                    'Battle royale game with ranking system',
                    'Fighting game with tournament support'
                ],
                'slightly_relevant': [
                    'Casual multiplayer game with some competitive elements',
                    'Co-op game with leaderboards',
                    'Party game with competitive mini-games'
                ],
                'irrelevant': [
                    'Single-player story adventure game',
                    'Relaxing puzzle game for casual play',
                    'Educational simulation without multiplayer'
                ]
            },
        
            # E-commerce scenarios (keeping existing ones)
            {
                'query': 'affordable mobile device for students',
                'highly_relevant': [
                    'Budget-friendly smartphone perfect for college students with essential features',
                    'Economical phone designed for young adults entering university',
                    'Cost-effective cellular device ideal for academic use and social media'
                ],
                'moderately_relevant': [
                    'Mid-range smartphone with good camera and battery life',
                    'Popular phone model used by many young professionals',
                    'Reliable mobile device with decent performance specifications'
                ],
                'slightly_relevant': [
                    'Premium flagship smartphone with advanced features',
                    'Professional tablet designed for business presentations',
                    'High-end laptop computer for gaming and development'
                ],
                'irrelevant': [
                    'Luxury watch collection with premium materials and craftsmanship',
                    'Organic skincare products for sensitive skin types',
                    'Professional kitchen appliances for gourmet cooking'
                ]
            },
        
            {
                'query': 'comfortable shoes for long walks',
                'highly_relevant': [
                    'Ergonomic walking sneakers with superior cushioning and arch support',
                    'Lightweight athletic footwear designed for extended hiking and trekking',
                    'Orthopedic shoes engineered for all-day comfort during extended periods'
                ],
                'moderately_relevant': [
                    'Casual sneakers with good padding and breathable materials',
                    'Running shoes suitable for jogging and light exercise',
                    'Versatile footwear appropriate for daily activities and commuting'
                ],
                'slightly_relevant': [
                    'Fashionable boots with stylish design and moderate comfort',
                    'Formal dress shoes for business and professional occasions',
                    'Specialized athletic cleats for sports and competitive activities'
                ],
                'irrelevant': [
                    'Luxury handbags and designer accessories for special events',
                    'High-performance laptop computers for software development',
                    'Gourmet coffee beans sourced from premium plantations'
                ]
            },
        
            # Tourism scenarios (keeping existing ones)
            {
                'query': 'beach vacation packages',
                'highly_relevant': [
                    'All-inclusive beach resort packages with ocean view accommodations',
                    'Tropical island vacation deals with water sports and beach activities',
                    'Coastal holiday packages featuring beachfront hotels and amenities'
                ],
                'moderately_relevant': [
                    'Seaside vacation rentals with beach access',
                    'Coastal city breaks with nearby beach attractions',
                    'Island hopping tours with beach destinations'
                ],
                'slightly_relevant': [
                    'Mountain resort packages with lake access',
                    'City vacation deals with outdoor activities',
                    'Adventure tours with some coastal elements'
                ],
                'irrelevant': [
                    'Mountain hiking expedition packages',
                    'Urban cultural tour experiences',
                    'Winter sports vacation deals'
                ]
            }
        ]
    
        # Add scenarios with different relevance levels
        for scenario in challenging_scenarios:
            query = scenario['query']
        
            # Highly relevant (score: 1.0)
            for doc in scenario['highly_relevant']:
                processed_samples.append({
                    'text': doc,
                    'query': query,
                    'source': 'synthetic_challenging',
                    'label': 'highly_relevant',
                    'relevance_score': 1.0
                })
        
            # Moderately relevant (score: 0.7)
            for doc in scenario['moderately_relevant']:
                processed_samples.append({
                    'text': doc,
                    'query': query,
                    'source': 'synthetic_challenging',
                    'label': 'moderately_relevant',
                    'relevance_score': 0.7
                })
        
            # Slightly relevant (score: 0.3)
            for doc in scenario['slightly_relevant']:
                processed_samples.append({
                    'text': doc,
                    'query': query,
                    'source': 'synthetic_challenging',
                    'label': 'slightly_relevant',
                    'relevance_score': 0.3
                })
        
            # Irrelevant (score: 0.0)
            for doc in scenario['irrelevant']:
                processed_samples.append({
                    'text': doc,
                    'query': query,
                    'source': 'synthetic_challenging',
                    'label': 'irrelevant',
                    'relevance_score': 0.0
                })
    
        # Process Steam games data
        if steam_data is not None and len(steam_data) > 0:
            print("Processing Steam games data...")
        
            # Create gaming-specific queries and documents
            gaming_queries = [
                'best indie games',
                'multiplayer action games',
                'RPG games with good story',
                'strategy games for beginners',
                'racing games realistic',
                'puzzle games challenging',
                'adventure games single player',
                'simulation games relaxing',
                'competitive esports games',
                'casual games for family'
            ]
        
            for _, row in steam_data.iterrows():
                # Create comprehensive text representation
                text_parts = []
            
                # Add title
                if 'title' in row and pd.notna(row['title']):
                    text_parts.append(str(row['title']))
            
                # Add genre
                if 'genre' in row and pd.notna(row['genre']):
                    text_parts.append(f"Genre: {row['genre']}")
            
                # Add description
                if 'description' in row and pd.notna(row['description']):
                    text_parts.append(str(row['description']))
            
                # Add tags if available
                if 'tags' in row and pd.notna(row['tags']):
                    text_parts.append(f"Tags: {row['tags']}")
            
                # Add developer if available
                if 'developer' in row and pd.notna(row['developer']):
                    text_parts.append(f"Developer: {row['developer']}")
            
                # Add platforms if available
                if 'platforms' in row and pd.notna(row['platforms']):
                    text_parts.append(f"Platforms: {row['platforms']}")
            
                if text_parts:
                    game_text = ' '.join(text_parts)
                
                    # Assign to relevant gaming queries based on genre/content
                    assigned_queries = []
                    game_text_lower = game_text.lower()
                
                    # Smart query assignment based on content
                    if any(word in game_text_lower for word in ['indie', 'independent']):
                        assigned_queries.append('best indie games')
                
                    if any(word in game_text_lower for word in ['multiplayer', 'online', 'pvp']):
                        assigned_queries.append('multiplayer action games')
                
                    if any(word in game_text_lower for word in ['rpg', 'role-playing', 'character']):
                        assigned_queries.append('RPG games with good story')
                
                    if any(word in game_text_lower for word in ['strategy', 'tactical', 'rts']):
                        assigned_queries.append('strategy games for beginners')
                
                    if any(word in game_text_lower for word in ['racing', 'driving', 'car']):
                        assigned_queries.append('racing games realistic')
                
                    if any(word in game_text_lower for word in ['puzzle', 'brain', 'logic']):
                        assigned_queries.append('puzzle games challenging')
                
                    if any(word in game_text_lower for word in ['adventure', 'exploration', 'story']):
                        assigned_queries.append('adventure games single player')
                
                    if any(word in game_text_lower for word in ['simulation', 'sim', 'realistic']):
                        assigned_queries.append('simulation games relaxing')
                
                    if any(word in game_text_lower for word in ['competitive', 'esports', 'tournament']):
                        assigned_queries.append('competitive esports games')
                
                    if any(word in game_text_lower for word in ['casual', 'family', 'easy']):
                        assigned_queries.append('casual games for family')
                
                    # If no specific assignment, use a general gaming query
                    if not assigned_queries:
                        assigned_queries = [np.random.choice(gaming_queries)]
                
                    # Calculate relevance based on review scores and content match
                    base_relevance = 0.6
                
                    # Adjust based on reviews if available
                    if 'positive_reviews' in row and 'negative_reviews' in row:
                        if pd.notna(row['positive_reviews']) and pd.notna(row['negative_reviews']):
                            total_reviews = row['positive_reviews'] + row['negative_reviews']
                            if total_reviews > 0:
                                positive_ratio = row['positive_reviews'] / total_reviews
                                base_relevance = 0.3 + (positive_ratio * 0.7)  # Scale to 0.3-1.0
                
                    # Add some noise for realistic variation
                    noise = np.random.normal(0, 0.1)
                    final_relevance = max(0, min(1, base_relevance + noise))
                
                    # Add to processed samples for each assigned query
                    for query in assigned_queries:
                        processed_samples.append({
                            'text': game_text,
                            'query': query,
                            'source': 'steam_games',
                            'label': 'relevant' if final_relevance > 0.5 else 'irrelevant',
                            'relevance_score': final_relevance
                        })
        
            print(f"✅ Processed {len(steam_data)} Steam games")
    
        # Process original tourism data if available
        if self.tourism_data is not None:
            for _, row in self.tourism_data.iterrows():
                text_features = []
                for col in self.tourism_data.columns:
                    if self.tourism_data[col].dtype == 'object' and pd.notna(row[col]):
                        text_features.append(str(row[col]))
            
            if text_features:
                processed_samples.append({
                    'text': ' '.join(text_features),
                    'query': 'tourism behavior analysis',
                    'source': 'tourism',
                    'label': 'relevant',
                    'relevance_score': 0.8
                })
    
        # Process e-commerce data with more nuanced relevance
        if self.ecommerce_data is not None:
            for _, row in self.ecommerce_data.iterrows():
                query = row['search_query']
                product_text = f"{row['product_name']} {row['category']} {row['review_text']}"
            
                # More nuanced relevance based on rating and query match
                base_relevance = 0.6 if row['rating'] >= 4 else 0.4
            
                # Add some noise to make it more challenging
                noise = np.random.normal(0, 0.1)
                relevance = max(0, min(1, base_relevance + noise))
            
                processed_samples.append({
                    'text': product_text,
                    'query': query,
                    'source': 'ecommerce',
                    'label': 'relevant' if relevance > 0.5 else 'irrelevant',
                    'relevance_score': relevance
                })
    
        self.processed_data = pd.DataFrame(processed_samples)
        print(f"Enhanced processed data created: {self.processed_data.shape}")
        print(f"Data sources: {self.processed_data['source'].value_counts().to_dict()}")
        print(f"Relevance score distribution:")
        print(self.processed_data['relevance_score'].describe())
        return self.processed_data

    def get_query_document_pairs(self):
        """Get query-document pairs with relevance scores for evaluation"""
        if self.processed_data is None:
            self.preprocess_data()
        
        query_doc_pairs = []
        queries = self.processed_data['query'].unique()
        
        for query in queries:
            query_data = self.processed_data[self.processed_data['query'] == query]
            documents = query_data['text'].tolist()
            relevance_scores = query_data['relevance_score'].tolist()
            
            # Shuffle documents to avoid any ordering bias
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
