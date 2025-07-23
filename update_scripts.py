#!/usr/bin/env python3
"""
Update all scripts to use the local-only data loader
"""

import os
import re

def update_file(file_path, old_import, new_import):
    """Update import statement in a file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace import statement
        updated_content = content.replace(old_import, new_import)
        
        if content == updated_content:
            print(f"⚠️ No changes needed in {file_path}")
            return False
        
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def update_all_scripts():
    """Update all scripts to use the local-only data loader"""
    print("🔄 UPDATING SCRIPTS TO USE LOCAL-ONLY DATA LOADER")
    print("=" * 60)
    
    files_to_update = [
        'interactive_query.py',
        'quick_query.py',
        'batch_query.py',
        'gaming_query.py',
        'main_comparison.py'
    ]
    
    old_import = "from data_loader import DataLoader"
    new_import = "from data_loader_local_only import DataLoader"
    
    updated_files = []
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            if update_file(file_path, old_import, new_import):
                updated_files.append(file_path)
    
    if updated_files:
        print(f"\n✅ Updated {len(updated_files)} files to use local-only data loader:")
        for file in updated_files:
            print(f"  • {file}")
        print("\nYou can now run these scripts without Kaggle downloads")
    else:
        print("\n⚠️ No files were updated")

if __name__ == "__main__":
    update_all_scripts()
