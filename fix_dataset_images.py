
import os
import shutil
from PIL import Image

def fix_dataset_images(dataset_path, output_path):
    """Fix common image issues that cause upload failures"""
    os.makedirs(output_path, exist_ok=True)
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        output_folder = os.path.join(output_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Processing {folder_name}...")
        fixed_count = 0
        failed_count = 0
        
        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            img_path = os.path.join(folder_path, img_name)
            output_path_img = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}.jpg")
            
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    
                    # Ensure minimum size
                    if min(img.size) < 256:
                        img = img.resize((max(256, img.width), max(256, img.height)), 
                                       Image.Resampling.LANCZOS)
                    
                    # Save as high-quality JPEG
                    img.save(output_path_img, 'JPEG', quality=90, optimize=True)
                    fixed_count += 1
                    
            except Exception as e:
                print(f"  Failed to fix {img_name}: {e}")
                failed_count += 1
        
        print(f"  Fixed: {fixed_count}, Failed: {failed_count}")

# Usage
# fix_dataset_images("path/to/original/dataset", "path/to/fixed/dataset")
