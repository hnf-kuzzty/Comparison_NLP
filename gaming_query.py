#!/usr/bin/env python3
"""
Gaming-Specific Query Interface
Specialized interface for querying Steam games data
"""

import sys
import pandas as pd
from interactive_query import InteractiveQuerySystem

class GamingQuerySystem(InteractiveQuerySystem):
    def __init__(self):
        super().__init__()
        self.steam_data = None
    
    def initialize_gaming_system(self):
        """Initialize system with focus on gaming data"""
        print("🎮 Initializing Gaming Query System...")
        print("=" * 60)
        
        # Load data with Steam games
        from data_loader import DataLoader
        data_loader = DataLoader()
        
        # Load Steam games specifically
        print("📥 Loading Steam games dataset...")
        self.steam_data = data_loader.load_steam_games_data()
        
        # Load other data
        tourism_data = data_loader.load_tourism_data()
        ecommerce_data = data_loader.load_ecommerce_data_sample()
        processed_data = data_loader.preprocess_data()
        
        if processed_data is None or len(processed_data) == 0:
            print("❌ Error: No data available. Cannot initialize system.")
            return False
        
        # Show data distribution
        print(f"\n📊 Data Distribution:")
        source_counts = processed_data['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} documents")
        
        # Extract documents and metadata
        self.documents = processed_data['text'].tolist()
        self.document_metadata = processed_data[['query', 'source', 'relevance_score']].to_dict('records')
        
        print(f"✅ Loaded {len(self.documents)} total documents")
        
        # Initialize models
        return super().initialize_system()
    
    def gaming_interactive_mode(self):
        """Gaming-focused interactive mode"""
        if not self.is_initialized:
            if not self.initialize_gaming_system():
                return
        
        print("\n🎮 GAMING QUERY MODE")
        print("=" * 60)
        print("Enter gaming-related queries to find relevant games and content.")
        print("\nSuggested gaming queries:")
        print("  - 'best RPG games with character customization'")
        print("  - 'multiplayer competitive games for esports'")
        print("  - 'indie games with unique mechanics'")
        print("  - 'strategy games for beginners'")
        print("  - 'racing games with realistic physics'")
        print("\nCommands:")
        print("  - 'gaming-stats' - Show gaming data statistics")
        print("  - 'popular-genres' - Show most common game genres")
        print("  - 'sample-games' - Show sample games in dataset")
        print("  - 'quit' - Exit")
        print()
        
        while True:
            try:
                query = input("🎮 Enter your gaming query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Happy gaming!")
                    break
                
                elif query.lower() == 'gaming-stats':
                    self.show_gaming_stats()
                    continue
                
                elif query.lower() == 'popular-genres':
                    self.show_popular_genres()
                    continue
                
                elif query.lower() == 'sample-games':
                    self.show_sample_games()
                    continue
                
                elif len(query.strip()) == 0:
                    print("Please enter a valid gaming query.")
                    continue
                
                # Process gaming query
                print(f"\n⏳ Searching for: '{query}'...")
                self.compare_models_for_query(query, top_k=5)
                
                # Show gaming-specific insights
                self.show_gaming_insights(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Happy gaming!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_gaming_stats(self):
        """Show statistics about gaming data"""
        if self.steam_data is not None:
            print(f"\n🎮 GAMING DATA STATISTICS")
            print("-" * 40)
            print(f"Total games: {len(self.steam_data)}")
            
            if 'genre' in self.steam_data.columns:
                print(f"Unique genres: {self.steam_data['genre'].nunique()}")
                print(f"Most common genres:")
                genre_counts = self.steam_data['genre'].value_counts().head(5)
                for genre, count in genre_counts.items():
                    print(f"  {genre}: {count} games")
            
            if 'positive_reviews' in self.steam_data.columns:
                avg_positive = self.steam_data['positive_reviews'].mean()
                print(f"Average positive reviews: {avg_positive:.0f}")
            
            if 'price' in self.steam_data.columns:
                avg_price = self.steam_data['price'].mean()
                print(f"Average price: ${avg_price:.2f}")
        else:
            print("No Steam games data available")
    
    def show_popular_genres(self):
        """Show popular game genres"""
        if self.steam_data is not None and 'genre' in self.steam_data.columns:
            print(f"\n🏆 POPULAR GAME GENRES")
            print("-" * 30)
            genre_counts = self.steam_data['genre'].value_counts().head(10)
            for i, (genre, count) in enumerate(genre_counts.items(), 1):
                print(f"{i:2d}. {genre:<15} ({count} games)")
        else:
            print("Genre information not available")
    
    def show_sample_games(self):
        """Show sample games from dataset"""
        if self.steam_data is not None:
            print(f"\n🎲 SAMPLE GAMES")
            print("-" * 30)
            sample_games = self.steam_data.sample(min(5, len(self.steam_data)))
            
            for _, game in sample_games.iterrows():
                title = game.get('title', 'Unknown Title')
                genre = game.get('genre', 'Unknown Genre')
                print(f"• {title} ({genre})")
                
                if 'description' in game and pd.notna(game['description']):
                    desc = str(game['description'])[:100] + "..." if len(str(game['description'])) > 100 else str(game['description'])
                    print(f"  {desc}")
                print()
        else:
            print("No games data available")
    
    def show_gaming_insights(self, query):
        """Show gaming-specific insights for the query"""
        gaming_docs = [i for i, meta in enumerate(self.document_metadata) if meta['source'] == 'steam_games']
        
        if gaming_docs:
            print(f"\n🎯 GAMING INSIGHTS FOR '{query}'")
            print("-" * 50)
            print(f"Found {len(gaming_docs)} gaming documents in corpus")
            
            # Show relevance distribution for gaming docs
            gaming_relevance = [self.document_metadata[i]['relevance_score'] for i in gaming_docs]
            if gaming_relevance:
                avg_relevance = sum(gaming_relevance) / len(gaming_relevance)
                print(f"Average gaming content relevance: {avg_relevance:.3f}")

def main():
    """Main function for gaming query system"""
    print("🎮 STEAM GAMES NLP QUERY SYSTEM")
    print("=" * 60)
    
    system = GamingQuerySystem()
    
    print("Choose an option:")
    print("1. Gaming interactive mode")
    print("2. Test with gaming sample queries")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            system.gaming_interactive_mode()
            break
        elif choice == '2':
            if system.initialize_gaming_system():
                test_gaming_queries(system)
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3")

def test_gaming_queries(system):
    """Test with gaming-specific queries"""
    gaming_queries = [
        "best RPG games with character customization",
        "multiplayer competitive games for esports",
        "indie games with unique art style",
        "strategy games for beginners",
        "racing games with realistic physics",
        "puzzle games challenging and creative",
        "adventure games single player story",
        "simulation games relaxing gameplay"
    ]
    
    print("\n🧪 TESTING GAMING QUERIES")
    print("=" * 60)
    
    for i, query in enumerate(gaming_queries, 1):
        print(f"\n🎮 Gaming Query {i}/{len(gaming_queries)}")
        system.compare_models_for_query(query, top_k=3)
        system.show_gaming_insights(query)
        
        if i < len(gaming_queries):
            input("\nPress Enter for next gaming query...")

if __name__ == "__main__":
    main()
