import os
from PIL import Image
import config

def diagnose_dataset():
    """Comprehensive dataset analysis"""
    print("🔍 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 60)
    
    if not os.path.exists(config.DATASET_PATH):
        print(f"❌ Dataset path doesn't exist: {config.DATASET_PATH}")
        return
    
    print(f"📁 Dataset path: {config.DATASET_PATH}")
    
    # Get all folders
    folders = []
    for item in os.listdir(config.DATASET_PATH):
        item_path = os.path.join(config.DATASET_PATH, item)
        if os.path.isdir(item_path):
            folders.append(item)
    
    folders.sort()
    print(f"📂 Found {len(folders)} folders: {folders}")
    
    # Analyze each folder
    total_images = 0
    folder_analysis = {}
    
    for folder_name in folders:
        folder_path = os.path.join(config.DATASET_PATH, folder_name)
        print(f"\n{'='*20} FOLDER: {folder_name} {'='*20}")
        
        # Get image files
        image_files = []
        all_files = os.listdir(folder_path)
        
        for file in all_files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(file)
        
        print(f"📊 Total files in folder: {len(all_files)}")
        print(f"🖼️  Image files: {len(image_files)}")
        
        if not image_files:
            print("⚠️  NO IMAGES FOUND!")
            continue
        
        # Analyze first 5 images
        valid_images = 0
        corrupted_images = 0
        
        for i, img_name in enumerate(image_files[:5]):
            img_path = os.path.join(folder_path, img_name)
            try:
                with Image.open(img_path) as img:
                    print(f"  ✓ {img_name}: {img.size}px, {img.mode}, {os.path.getsize(img_path)} bytes")
                    valid_images += 1
            except Exception as e:
                print(f"  ❌ {img_name}: CORRUPTED - {e}")
                corrupted_images += 1
        
        # Check all images for corruption
        for img_name in image_files[5:]:
            img_path = os.path.join(folder_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check if image is corrupted
                valid_images += 1
            except:
                corrupted_images += 1
        
        folder_analysis[folder_name] = {
            'total_images': len(image_files),
            'valid_images': valid_images,
            'corrupted_images': corrupted_images
        }
        
        total_images += len(image_files)
        
        print(f"✅ Valid images: {valid_images}")
        print(f"❌ Corrupted images: {corrupted_images}")
        
        if len(image_files) < 5:
            print("⚠️  WARNING: Less than 5 images (minimum required)")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    print(f"Total folders: {len(folders)}")
    print(f"Total images: {total_images}")
    
    print(f"\n📊 Per-folder breakdown:")
    for folder, stats in folder_analysis.items():
        status = "✅" if stats['valid_images'] >= 5 else "⚠️"
        print(f"  {status} {folder}: {stats['valid_images']} valid, {stats['corrupted_images']} corrupted")
    
    # Check for common issues
    print(f"\n🔍 POTENTIAL ISSUES:")
    
    # Issue 1: Folders with < 5 images
    low_image_folders = [f for f, s in folder_analysis.items() if s['valid_images'] < 5]
    if low_image_folders:
        print(f"  ❌ Folders with <5 images: {low_image_folders}")
    
    # Issue 2: Corrupted images
    corrupted_folders = [f for f, s in folder_analysis.items() if s['corrupted_images'] > 0]
    if corrupted_folders:
        print(f"  ❌ Folders with corrupted images: {corrupted_folders}")
    
    # Issue 3: Case sensitivity issues
    if any(folder.islower() for folder in folders) and any(folder.isupper() for folder in folders):
        print(f"  ⚠️  Mixed case folder names detected")
    
    # Issue 4: Unexpected folder names
    expected_letters = set('abcdefghijklmnopqrstuvwxyz')
    folder_letters = set(folder.lower() for folder in folders)
    unexpected = folder_letters - expected_letters
    if unexpected:
        print(f"  ⚠️  Unexpected folder names: {unexpected}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if low_image_folders:
        print(f"  1. Add more images to: {low_image_folders}")
    
    if corrupted_folders:
        print(f"  2. Fix corrupted images in: {corrupted_folders}")
    
    if total_images < 130:  # Less than 5 per letter
        print(f"  3. Dataset seems small - consider adding more images")
    
    print(f"  4. Verify folder names match ASL letters exactly")

if __name__ == "__main__":
    diagnose_dataset()