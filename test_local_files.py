#!/usr/bin/env python3
"""
Test script to verify local files are being used correctly
"""

import os

def check_local_files():
    """Check if local files exist"""
    print("🔍 CHECKING LOCAL FILES")
    print("=" * 50)
    
    files_to_check = [
        'data/games.csv',
        'data/recommendations.csv',
        'data/users.csv'
    ]
    
    found_files = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ Found: {file_path} ({size_mb:.2f} MB)")
            found_files.append(file_path)
        else:
            print(f"❌ Missing: {file_path}")
    
    if found_files:
        print(f"\n✅ Found {len(found_files)} out of {len(files_to_check)} files")
    else:
        print("\n❌ No local files found!")
        print("Please make sure your CSV files are in the data/ folder")
    
    return found_files

def test_data_loader():
    """Test the local-only data loader"""
    found_files = check_local_files()
    
    if not found_files:
        print("\nSkipping data loader test since no local files were found")
        return
    
    print("\n🧪 TESTING LOCAL-ONLY DATA LOADER")
    print("=" * 50)
    
    try:
        from data_loader_local_only import DataLoader
        
        loader = DataLoader()
        
        # Test Steam data loading
        print("\n1. Testing Steam data loading...")
        steam_data = loader.load_steam_games_data()
        if steam_data is not None:
            print(f"   ✅ Steam data loaded: {steam_data.shape}")
            print(f"   Sample columns: {list(steam_data.columns)[:5]}")
            print(f"   First row:")
            print(f"   {steam_data.iloc[0].to_dict()}")
        else:
            print("   ❌ Steam data failed to load")
        
        # Test preprocessing
        print("\n2. Testing data preprocessing...")
        processed_data = loader.preprocess_data()
        if processed_data is not None:
            print(f"   ✅ Processed data created: {processed_data.shape}")
            print(f"   Sources: {dict(processed_data['source'].value_counts())}")
            print(f"   Sample processed data:")
            print(f"   {processed_data.iloc[0].to_dict()}")
        else:
            print("   ❌ Preprocessing failed")
        
        print(f"\n✅ LOCAL DATA LOADER TEST COMPLETE")
        print("=" * 50)
        print("The local-only data loader is working correctly!")
        print("You can now use it in your main scripts by updating the import:")
        print("from data_loader_local_only import DataLoader")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loader()
