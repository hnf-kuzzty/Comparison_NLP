#!/usr/bin/env python3
"""
Test script to verify data loader works correctly
"""

def test_data_loader():
    print("🧪 TESTING DATA LOADER")
    print("=" * 50)
    
    try:
        # Test the simple version first
        print("Testing simple data loader...")
        from data_loader_simple import DataLoader
        
        loader = DataLoader()
        
        # Test Steam data loading
        print("\n1. Testing Steam data loading...")
        steam_data = loader.load_steam_games_data()
        if steam_data is not None:
            print(f"   ✅ Steam data loaded: {steam_data.shape}")
            print(f"   Columns: {list(steam_data.columns)}")
        else:
            print("   ❌ Steam data failed to load")
        
        # Test e-commerce data
        print("\n2. Testing e-commerce data...")
        ecommerce_data = loader.load_ecommerce_data_sample()
        if ecommerce_data is not None:
            print(f"   ✅ E-commerce data created: {ecommerce_data.shape}")
        else:
            print("   ❌ E-commerce data failed")
        
        # Test preprocessing
        print("\n3. Testing data preprocessing...")
        processed_data = loader.preprocess_data()
        if processed_data is not None:
            print(f"   ✅ Processed data created: {processed_data.shape}")
            print(f"   Sources: {dict(processed_data['source'].value_counts())}")
        else:
            print("   ❌ Preprocessing failed")
        
        # Test query pairs
        print("\n4. Testing query-document pairs...")
        query_pairs = loader.get_query_document_pairs()
        if query_pairs:
            print(f"   ✅ Query pairs created: {len(query_pairs)}")
            print(f"   Sample query: {query_pairs[0]['query']}")
            print(f"   Documents for query: {len(query_pairs[0]['documents'])}")
        else:
            print("   ❌ Query pairs failed")
        
        print(f"\n✅ DATA LOADER TEST COMPLETE")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loader()
