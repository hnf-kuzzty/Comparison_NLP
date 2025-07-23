#!/usr/bin/env python3
"""
Steam Data Inspector
Analyze your local Steam CSV files to understand their structure
"""

import pandas as pd
import os

def inspect_steam_files():
    """Inspect the three Steam CSV files"""
    print("🔍 STEAM DATA INSPECTOR")
    print("=" * 60)
    
    files_to_check = {
        'games': 'data/games.csv',
        'recommendations': 'data/recommendations.csv',
        'users': 'data/users.csv'
    }
    
    file_info = {}
    
    for file_type, file_path in files_to_check.items():
        print(f"\n📁 Inspecting {file_type}.csv...")
        print("-" * 40)
        
        if os.path.exists(file_path):
            try:
                # Load the file
                df = pd.read_csv(file_path)
                file_info[file_type] = df
                
                print(f"✅ File found and loaded successfully")
                print(f"📊 Shape: {df.shape} (rows × columns)")
                print(f"📋 Columns ({len(df.columns)}):")
                
                for i, col in enumerate(df.columns, 1):
                    dtype = df[col].dtype
                    non_null = df[col].count()
                    null_count = df[col].isnull().sum()
                    print(f"  {i:2d}. {col:<25} | {dtype:<10} | {non_null:>6} non-null | {null_count:>6} null")
                
                # Show sample data
                print(f"\n📝 Sample data (first 3 rows):")
                print(df.head(3).to_string())
                
                # Show data types and memory usage
                print(f"\n💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                
                # Show unique values for key columns
                key_columns = ['genre', 'genres', 'category', 'categories', 'developer', 'publisher']
                for col in key_columns:
                    if col in df.columns:
                        unique_count = df[col].nunique()
                        print(f"🔑 {col}: {unique_count} unique values")
                        if unique_count <= 10:
                            print(f"   Values: {list(df[col].unique())}")
                        else:
                            print(f"   Top 5: {list(df[col].value_counts().head().index)}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                
        else:
            print(f"❌ File not found: {file_path}")
    
    # Cross-file analysis
    if len(file_info) > 1:
        print(f"\n🔗 CROSS-FILE ANALYSIS")
        print("=" * 40)
        
        # Check for common columns that could be used for merging
        if 'games' in file_info and 'recommendations' in file_info:
            games_cols = set(file_info['games'].columns)
            rec_cols = set(file_info['recommendations'].columns)
            common_cols = games_cols & rec_cols
            
            print(f"📋 Games ↔ Recommendations common columns: {common_cols}")
            
            # Check if we can merge on these columns
            for col in common_cols:
                games_unique = file_info['games'][col].nunique()
                rec_unique = file_info['recommendations'][col].nunique()
                print(f"  {col}: Games({games_unique}) ↔ Recommendations({rec_unique})")
        
        if 'users' in file_info and 'recommendations' in file_info:
            users_cols = set(file_info['users'].columns)
            rec_cols = set(file_info['recommendations'].columns)
            common_cols = users_cols & rec_cols
            
            print(f"📋 Users ↔ Recommendations common columns: {common_cols}")
    
    # Generate recommendations for data usage
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 40)
    
    if 'games' in file_info:
        games_df = file_info['games']
        
        # Check for text-rich columns
        text_columns = []
        for col in games_df.columns:
            if games_df[col].dtype == 'object':
                avg_length = games_df[col].astype(str).str.len().mean()
                if avg_length > 20:  # Likely text content
                    text_columns.append((col, avg_length))
        
        if text_columns:
            print("📝 Best columns for text analysis:")
            for col, avg_len in sorted(text_columns, key=lambda x: x[1], reverse=True):
                print(f"  • {col} (avg length: {avg_len:.1f} chars)")
        
        # Check for rating/review columns
        rating_columns = [col for col in games_df.columns 
                         if any(word in col.lower() for word in ['rating', 'review', 'score', 'positive', 'negative'])]
        if rating_columns:
            print(f"⭐ Rating/Review columns: {rating_columns}")
        
        # Check for categorical columns
        categorical_columns = []
        for col in games_df.columns:
            if games_df[col].dtype == 'object' and games_df[col].nunique() < 50:
                categorical_columns.append((col, games_df[col].nunique()))
        
        if categorical_columns:
            print("🏷️  Good categorical columns for grouping:")
            for col, unique_count in categorical_columns:
                print(f"  • {col} ({unique_count} categories)")
    
    return file_info

def suggest_query_scenarios(file_info):
    """Suggest query scenarios based on available data"""
    print(f"\n🎯 SUGGESTED QUERY SCENARIOS")
    print("=" * 40)
    
    if 'games' in file_info:
        games_df = file_info['games']
        
        # Genre-based queries
        if 'genre' in games_df.columns or 'genres' in games_df.columns:
            genre_col = 'genre' if 'genre' in games_df.columns else 'genres'
            top_genres = games_df[genre_col].value_counts().head(5).index.tolist()
            print("🎮 Genre-based queries:")
            for genre in top_genres:
                print(f"  • 'best {genre.lower()} games'")
        
        # Developer-based queries
        if 'developer' in games_df.columns:
            top_devs = games_df['developer'].value_counts().head(3).index.tolist()
            print("\n🏢 Developer-based queries:")
            for dev in top_devs:
                print(f"  • 'games by {dev}'")
        
        # Price-based queries
        if 'price' in games_df.columns:
            print("\n💰 Price-based queries:")
            print("  • 'free games'")
            print("  • 'games under 10 dollars'")
            print("  • 'budget games with good reviews'")
        
        # Rating-based queries
        rating_cols = [col for col in games_df.columns 
                      if any(word in col.lower() for word in ['rating', 'review', 'positive'])]
        if rating_cols:
            print("\n⭐ Rating-based queries:")
            print("  • 'highly rated indie games'")
            print("  • 'games with positive reviews'")
            print("  • 'top rated strategy games'")

def main():
    """Main inspection function"""
    file_info = inspect_steam_files()
    
    if file_info:
        suggest_query_scenarios(file_info)
        
        print(f"\n✅ INSPECTION COMPLETE")
        print("=" * 40)
        print("Your data is ready to use! Run the updated data_loader.py")
        print("The system will automatically use your local files.")
    else:
        print(f"\n❌ NO FILES FOUND")
        print("=" * 40)
        print("Make sure your CSV files are in the data/ folder:")
        print("  • data/games.csv")
        print("  • data/recommendations.csv") 
        print("  • data/users.csv")

if __name__ == "__main__":
    main()
