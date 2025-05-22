import os
import time
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import config

class ASLModelTrainer:
    def __init__(self):
        self.training_key = config.TRAINING_KEY
        self.endpoint = config.ENDPOINT
        
        # Initialize Custom Vision Training Client
        credentials = ApiKeyCredentials(in_headers={"Training-key": self.training_key})
        self.trainer = CustomVisionTrainingClient(self.endpoint, credentials)
        self.project = None
        
    def get_project(self, project_name="ASL_Gesture_Recognition"):
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
        """Create tags for the dataset folders we actually have"""
        print("\n📋 Setting up tags...")
        
        # Get existing tags
        existing_tags = self.trainer.get_tags(self.project.id)
        existing_tag_names = {tag.name: tag for tag in existing_tags}
        
        # Determine which folders exist in our dataset
        dataset_folders = []
        if os.path.exists(config.DATASET_PATH):
            for item in os.listdir(config.DATASET_PATH):
                item_path = os.path.join(config.DATASET_PATH, item)
                if os.path.isdir(item_path):
                    dataset_folders.append(item)
        
        dataset_folders.sort()  # Sort for consistent output
        print(f"Found dataset folders: {dataset_folders}")
        
        # Create tags for existing folders only
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
        
    def upload_images(self, tags):
        """Upload images from dataset folders"""
        print(f"\n📁 Uploading images from: {config.DATASET_PATH}")
        
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
            
            print(f"  Found {len(image_files)} images")
            
            # Check existing images
            existing_images = self.trainer.get_tagged_images(self.project.id, tag_ids=[tag.id])
            if len(existing_images) >= len(image_files):
                print(f"  ℹ️ Tag '{folder_name}' already has {len(existing_images)} images. Skipping.")
                continue
            
            # Prepare images for upload
            image_list = []
            for image_name in image_files[:64]:  # Limit to 64 per batch
                image_path = os.path.join(folder_path, image_name)
                
                try:
                    with open(image_path, "rb") as image_contents:
                        image_data = image_contents.read()
                        image_list.append(ImageFileCreateEntry(
                            name=f"{folder_name}_{image_name}",
                            contents=image_data,
                            tag_ids=[tag.id]
                        ))
                except Exception as e:
                    print(f"    ❌ Error reading {image_name}: {e}")
            
            # Upload batch
            if image_list:
                try:
                    print(f"  📤 Uploading {len(image_list)} images...")
                    upload_result = self.trainer.create_images_from_files(
                        self.project.id, 
                        ImageFileCreateBatch(images=image_list)
                    )
                    
                    if upload_result.is_batch_successful:
                        total_uploaded += len(image_list)
                        print(f"  ✓ Successfully uploaded {len(image_list)} images")
                    else:
                        print(f"  ❌ Batch upload failed for {folder_name}")
                        
                except Exception as e:
                    print(f"  ❌ Error uploading batch: {e}")
        
        print(f"\n✅ Upload complete! Total new images uploaded: {total_uploaded}")
        
        # Show final image counts
        print("\n📊 Final image counts per tag:")
        for folder_name, tag in tags.items():
            images = self.trainer.get_tagged_images(self.project.id, tag_ids=[tag.id])
            print(f"  {folder_name}: {len(images)} images")
        
        return total_uploaded > 0 or any(
            len(self.trainer.get_tagged_images(self.project.id, tag_ids=[tag.id])) > 0 
            for tag in tags.values()
        )
    
    def train_model(self):
        """Train the Custom Vision model"""
        print("\n🚀 Starting model training...")
        
        try:
            # Check minimum requirements
            tags = self.trainer.get_tags(self.project.id)
            valid_tags = 0
            
            for tag in tags:
                images = self.trainer.get_tagged_images(self.project.id, tag_ids=[tag.id])
                if len(images) >= 5:
                    valid_tags += 1
                else:
                    print(f"  ⚠️ Tag '{tag.name}' has only {len(images)} images (minimum 5 required)")
            
            if valid_tags < 2:
                print(f"❌ Need at least 2 tags with 5+ images each. Found {valid_tags} valid tags.")
                return None
            
            print(f"✓ Found {valid_tags} tags with sufficient images")
            print("🎯 Training may take 10-30 minutes...")
            
            iteration = self.trainer.train_project(self.project.id)
            print(f"✓ Training started. Iteration ID: {iteration.id}")
            
            # Wait for training to complete
            start_time = time.time()
            while iteration.status != "Completed":
                iteration = self.trainer.get_iteration(self.project.id, iteration.id)
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                print(f"  Status: {iteration.status} (Elapsed: {mins}m {secs}s)")
                
                if iteration.status == "Failed":
                    print("❌ Training failed!")
                    return None
                    
                time.sleep(30)  # Check every 30 seconds
                
            print("🎉 Training completed successfully!")
            return iteration
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return None

def main():
    print("🤖 ASL Gesture Recognition - Clean Training")
    print("=" * 50)
    
    # Validate configuration
    try:
        if not hasattr(config, 'TRAINING_KEY') or not config.TRAINING_KEY:
            print("❌ TRAINING_KEY not found in config.py")
            return
        
        if not hasattr(config, 'ENDPOINT') or not config.ENDPOINT:
            print("❌ ENDPOINT not found in config.py")
            return
            
        if not hasattr(config, 'DATASET_PATH') or not os.path.exists(config.DATASET_PATH):
            print(f"❌ DATASET_PATH not found or doesn't exist: {getattr(config, 'DATASET_PATH', 'Not set')}")
            return
            
    except ImportError:
        print("❌ Could not import config.py - make sure it exists!")
        return
    
    trainer = ASLModelTrainer()
    
    try:
        # Step 1: Get/create project
        project_id = trainer.get_project()
        print(f"\n📝 Project ID: {project_id}")
        
        # Step 2: Setup tags based on actual dataset folders
        tags = trainer.setup_tags()
        if not tags:
            print("❌ No tags created - check your dataset folder")
            return
        
        # Step 3: Upload images
        if not trainer.upload_images(tags):
            print("❌ No images uploaded")
            return
        
        # Step 4: Train model
        iteration = trainer.train_model()
        if not iteration:
            print("❌ Training failed")
            return
        
        print("\n🎉 Training completed successfully!")
        print("=" * 50)
        print(f"✓ Project ID: {project_id}")
        print(f"✓ Iteration ID: {iteration.id}")
        print("\nNext steps:")
        print("1. Test your model in the Azure portal")
        print("2. Publish the iteration when ready")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()