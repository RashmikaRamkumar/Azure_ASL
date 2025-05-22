import os
import time
import random
from PIL import Image, ImageEnhance, ImageOps
import io
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import config

class ImprovedASLModelTrainer:
    def __init__(self):
        self.training_key = config.TRAINING_KEY
        self.endpoint = config.ENDPOINT
        
        # Initialize Custom Vision Training Client
        credentials = ApiKeyCredentials(in_headers={"Training-key": self.training_key})
        self.trainer = CustomVisionTrainingClient(self.endpoint, credentials)
        self.project = None
        
    def augment_image(self, image_path, num_augmentations=2):
        """Create augmented versions of an image to increase dataset diversity"""
        augmented_images = []
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to standard size first
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Original image
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=90)
                img_byte_arr.seek(0)
                augmented_images.append(img_byte_arr.getvalue())
                
                # Create augmentations
                for i in range(num_augmentations):
                    augmented = img.copy()
                    
                    # Random rotation (-10 to +10 degrees) - reduced range
                    if random.random() > 0.4:
                        angle = random.uniform(-10, 10)
                        augmented = augmented.rotate(angle, fillcolor=(255, 255, 255))
                    
                    # Random brightness adjustment - smaller range
                    if random.random() > 0.4:
                        enhancer = ImageEnhance.Brightness(augmented)
                        factor = random.uniform(0.9, 1.1)
                        augmented = enhancer.enhance(factor)
                    
                    # Random contrast adjustment - smaller range
                    if random.random() > 0.4:
                        enhancer = ImageEnhance.Contrast(augmented)
                        factor = random.uniform(0.9, 1.1)
                        augmented = enhancer.enhance(factor)
                    
                    # Random horizontal flip (30% chance) - reduced probability
                    if random.random() > 0.7:
                        augmented = ImageOps.mirror(augmented)
                    
                    # Ensure final size
                    augmented = augmented.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Convert to bytes with error checking
                    img_byte_arr = io.BytesIO()
                    augmented.save(img_byte_arr, format='JPEG', quality=90)
                    img_byte_arr.seek(0)
                    
                    # Validate image size (Azure has limits)
                    img_data = img_byte_arr.getvalue()
                    if len(img_data) > 4 * 1024 * 1024:  # 4MB limit
                        print(f"    ⚠️ Image too large: {len(img_data)} bytes, skipping")
                        continue
                    
                    augmented_images.append(img_data)
                    
        except Exception as e:
            print(f"    ❌ Error augmenting image {image_path}: {e}")
            # Return at least the original if possible
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG', quality=90)
                    augmented_images.append(img_byte_arr.getvalue())
            except:
                pass
            
        return augmented_images
    
    def get_project(self, project_name="ASL_Enhanced_Recognition"):
        """Get existing project or create new one"""
        print(f"Looking for project: {project_name}...")
        
        # Check if project already exists
        projects = self.trainer.get_projects()
        for project in projects:
            if project.name == project_name:
                print(f"✓ Found existing project: {project_name}")
                self.project = project
                return project.id
        
        # Create new project if it doesn't exist
        print(f"Creating new project: {project_name}...")
        self.project = self.trainer.create_project(project_name)
        print(f"✓ Project created successfully!")
        print(f"  Project ID: {self.project.id}")
        return self.project.id
    
    def setup_tags(self):
        """Create tags for the dataset folders"""
        print("\n📋 Setting up tags...")
        
        # Get existing tags
        existing_tags = self.trainer.get_tags(self.project.id)
        existing_tag_names = {tag.name: tag for tag in existing_tags}
        
        # Get dataset folders
        dataset_folders = []
        if os.path.exists(config.DATASET_PATH):
            for item in os.listdir(config.DATASET_PATH):
                item_path = os.path.join(config.DATASET_PATH, item)
                if os.path.isdir(item_path):
                    # Count images in folder
                    image_files = [f for f in os.listdir(item_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    if len(image_files) >= 5:  # Only include folders with enough images
                        dataset_folders.append(item)
                    else:
                        print(f"  ⚠️ Skipping folder '{item}' - only {len(image_files)} images")
        
        dataset_folders.sort()
        print(f"Valid dataset folders: {dataset_folders}")
        
        # Create tags
        tags = {}
        for folder_name in dataset_folders:
            if folder_name in existing_tag_names:
                tags[folder_name] = existing_tag_names[folder_name]
                print(f"  ✓ Using existing tag: {folder_name}")
            else:
                try:
                    tag = self.trainer.create_tag(self.project.id, folder_name)
                    tags[folder_name] = tag
                    print(f"  ✓ Created new tag: {folder_name}")
                except Exception as e:
                    print(f"  ❌ Failed to create tag '{folder_name}': {e}")
        
        print(f"✓ Successfully set up {len(tags)} tags")
        return tags
    
    def upload_images_with_augmentation(self, tags, augmentation_factor=2):
        """Upload images with data augmentation"""
        print(f"\n📁 Uploading images with augmentation from: {config.DATASET_PATH}")
        print(f"🔄 Augmentation factor: {augmentation_factor}x per image")
        
        if not os.path.exists(config.DATASET_PATH):
            print(f"❌ Dataset path does not exist: {config.DATASET_PATH}")
            return False
            
        total_uploaded = 0
        
        for folder_name, tag in tags.items():
            folder_path = os.path.join(config.DATASET_PATH, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
                
            print(f"\n📂 Processing folder: {folder_name}")
            
            # Get all image files
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not image_files:
                print(f"  ⚠️ No images found in {folder_name}")
                continue
            
            print(f"  Found {len(image_files)} original images")
            
            # Check existing images to avoid duplicates
            existing_images = self.trainer.get_tagged_images(self.project.id, tag_ids=[tag.id])
            expected_total = len(image_files) * (augmentation_factor + 1)  # +1 for original
            
            if len(existing_images) >= expected_total * 0.8:  # Allow some tolerance
                print(f"  ℹ️ Tag '{folder_name}' already has {len(existing_images)} images. Skipping.")
                continue
            
            # Process fewer images at a time to avoid quota issues
            batch_size = 8  # Reduced batch size
            image_batch = []
            batch_count = 0
            successful_uploads = 0
            
            # Limit to first 20 images per folder to avoid quota issues
            image_files = image_files[:20]
            random.shuffle(image_files)
            
            for image_name in image_files:
                image_path = os.path.join(folder_path, image_name)
                
                try:
                    # Generate augmented versions
                    augmented_images = self.augment_image(image_path, augmentation_factor)
                    
                    if not augmented_images:
                        print(f"    ⚠️ No valid images generated for {image_name}")
                        continue
                    
                    for idx, img_data in enumerate(augmented_images):
                        if not img_data or len(img_data) == 0:
                            continue
                            
                        suffix = "orig" if idx == 0 else f"aug{idx}"
                        unique_name = f"{folder_name}_{os.path.splitext(image_name)[0]}_{suffix}_{int(time.time()*1000) % 10000}"
                        
                        image_batch.append(ImageFileCreateEntry(
                            name=unique_name,
                            contents=img_data,
                            tag_ids=[tag.id]
                        ))
                        
                        # Upload when batch is full
                        if len(image_batch) >= batch_size:
                            success_count = self._upload_batch(image_batch, folder_name, batch_count)
                            successful_uploads += success_count
                            total_uploaded += success_count
                            image_batch = []
                            batch_count += 1
                            time.sleep(2)  # Longer delay between batches
                            
                except Exception as e:
                    print(f"    ❌ Error processing {image_name}: {e}")
            
            # Upload remaining images in batch
            if image_batch:
                success_count = self._upload_batch(image_batch, folder_name, batch_count)
                successful_uploads += success_count
                total_uploaded += success_count
            
            print(f"  ✅ Successfully uploaded {successful_uploads} images for {folder_name}")
        
        print(f"\n✅ Upload complete! Total images uploaded: {total_uploaded}")
        
        # Show final counts
        print("\n📊 Final image counts per tag:")
        for folder_name, tag in tags.items():
            images = self.trainer.get_tagged_images(self.project.id, tag_ids=[tag.id])
            print(f"  {folder_name}: {len(images)} images")
        
        return total_uploaded > 0
    
    def _upload_batch(self, image_batch, folder_name, batch_count):
        """Upload a batch of images with detailed error handling"""
        try:
            print(f"    📤 Uploading batch {batch_count + 1} with {len(image_batch)} images...")
            
            upload_result = self.trainer.create_images_from_files(
                self.project.id, 
                ImageFileCreateBatch(images=image_batch)
            )
            
            successful_count = 0
            failed_count = 0
            
            if upload_result.images:
                for img_result in upload_result.images:
                    if hasattr(img_result, 'status'):
                        if img_result.status == "OK" or img_result.status == "OKDuplicate":
                            successful_count += 1
                        else:
                            failed_count += 1
                            if hasattr(img_result, 'status_description'):
                                print(f"      ❌ Image failed: {img_result.status} - {img_result.status_description}")
                            else:
                                print(f"      ❌ Image failed: {img_result.status}")
                    else:
                        successful_count += 1  # Assume success if no status
            
            if upload_result.is_batch_successful or successful_count > 0:
                print(f"    ✓ Batch {batch_count + 1}: {successful_count} uploaded, {failed_count} failed")
                return successful_count
            else:
                print(f"    ❌ Batch {batch_count + 1}: All {len(image_batch)} images failed")
                return 0
                
        except Exception as e:
            print(f"    ❌ Error uploading batch {batch_count + 1}: {e}")
            print(f"    📝 Exception details: {type(e).__name__}")
            return 0
    
    def train_model(self):
        """Train the Custom Vision model with better parameters"""
        print("\n🚀 Starting enhanced model training...")
        
        try:
            # Validate dataset
            tags = self.trainer.get_tags(self.project.id)
            valid_tags = 0
            total_images = 0
            
            print("\n📊 Dataset validation:")
            for tag in tags:
                images = self.trainer.get_tagged_images(self.project.id, tag_ids=[tag.id])
                total_images += len(images)
                if len(images) >= 5:  # Lower minimum for testing
                    valid_tags += 1
                    print(f"  ✓ {tag.name}: {len(images)} images")
                else:
                    print(f"  ⚠️ {tag.name}: {len(images)} images (need at least 5)")
            
            if valid_tags < 2:
                print(f"❌ Need at least 2 tags with 5+ images each. Found {valid_tags} valid tags.")
                return None
            
            if total_images < 50:
                print(f"⚠️ Total images ({total_images}) is quite low. Training may not be optimal.")
            
            print(f"\n✓ Dataset ready: {valid_tags} classes, {total_images} total images")
            print("🎯 Training will take 10-30 minutes depending on dataset size...")
            
            # Start training
            iteration = self.trainer.train_project(self.project.id)
            print(f"✓ Training started. Iteration ID: {iteration.id}")
            
            # Monitor training progress
            start_time = time.time()
            last_status = ""
            
            while iteration.status not in ["Completed", "Failed"]:
                iteration = self.trainer.get_iteration(self.project.id, iteration.id)
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                
                if iteration.status != last_status:
                    print(f"  Status: {iteration.status} (Elapsed: {mins}m {secs}s)")
                    last_status = iteration.status
                elif elapsed % 120 == 0:  # Print every 2 minutes
                    print(f"  Still training... (Elapsed: {mins}m {secs}s)")
                
                time.sleep(30)
            
            if iteration.status == "Completed":
                print("🎉 Training completed successfully!")
                
                # Get performance metrics if available
                try:
                    performance = self.trainer.get_iteration_performance(self.project.id, iteration.id)
                    print(f"\n📈 Model Performance:")
                    print(f"  Overall Precision: {performance.precision:.3f}")
                    print(f"  Overall Recall: {performance.recall:.3f}")
                    print(f"  Average Precision: {performance.average_precision:.3f}")
                    
                    if hasattr(performance, 'per_tag_performance') and performance.per_tag_performance:
                        print(f"\n📊 Per-class Performance:")
                        for tag_perf in performance.per_tag_performance:
                            print(f"  {tag_perf.name}: P={tag_perf.precision:.3f}, R={tag_perf.recall:.3f}")
                            
                except Exception as e:
                    print(f"  ℹ️ Performance metrics not yet available: {e}")
                
                return iteration
            else:
                print("❌ Training failed!")
                return None
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return None

def main():
    print("🚀 Enhanced ASL Gesture Recognition Training")
    print("=" * 60)
    
    # Validate configuration
    try:
        required_attrs = ['TRAINING_KEY', 'ENDPOINT', 'DATASET_PATH']
        for attr in required_attrs:
            if not hasattr(config, attr) or not getattr(config, attr):
                print(f"❌ {attr} not found in config.py")
                return
                
        if not os.path.exists(config.DATASET_PATH):
            print(f"❌ Dataset path doesn't exist: {config.DATASET_PATH}")
            return
            
    except ImportError:
        print("❌ Could not import config.py")
        return
    
    trainer = ImprovedASLModelTrainer()
    
    try:
        # Step 1: Setup project
        project_id = trainer.get_project()
        print(f"\n📝 Project ID: {project_id}")
        
        # Step 2: Setup tags
        tags = trainer.setup_tags()
        if not tags:
            print("❌ No valid tags found")
            return
        
        # Step 3: Upload with augmentation (reduced factor)
        print(f"\n🔄 Starting data augmentation and upload...")
        augmentation_factor = 2  # Reduced from 3 to 2
        
        if not trainer.upload_images_with_augmentation(tags, augmentation_factor):
            print("❌ Image upload failed")
            return
        
        # Step 4: Train model
        iteration = trainer.train_model()
        if not iteration:
            print("❌ Training failed")
            return
        
        print("\n🎉 Training pipeline completed successfully!")
        print("=" * 60)
        print(f"✓ Project ID: {project_id}")
        print(f"✓ Iteration ID: {iteration.id}")
        
        print("\n📋 Next Steps:")
        print("1. Test your model in the Azure Custom Vision portal")
        print("2. Review per-class performance metrics")
        print("3. If performance is still low:")
        print("   - Add more diverse images to underperforming classes")
        print("   - Ensure hand gestures are clearly visible and well-lit")
        print("   - Consider different backgrounds and hand positions")
        print("4. Publish the iteration when satisfied with performance")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()