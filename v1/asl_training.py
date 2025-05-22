import os
import time
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials

class ASLModelTrainer:
    def __init__(self, training_key, endpoint):
        self.training_key = training_key
        self.endpoint = endpoint
        self.trainer = CustomVisionTrainingClient(training_key, endpoint=endpoint)
        self.project = None
        
    def create_project(self, project_name="ASL Gesture Recognition"):
        """Create a new Custom Vision project"""
        print("Creating project...")
        self.project = self.trainer.create_project(project_name)
        print(f"Project created with id: {self.project.id}")
        return self.project.id
        
    def create_tags(self):
        """Create tags for numbers (0-9) and letters (A-Z)"""
        print("Creating tags...")
        tags = {}
        
        # Create number tags (0-9)
        for i in range(10):
            tag = self.trainer.create_tag(self.project.id, str(i))
            tags[str(i)] = tag
            print(f"Created tag: {i}")
            
        # Create letter tags (A-Z)
        for i in range(26):
            letter = chr(ord('A') + i)
            tag = self.trainer.create_tag(self.project.id, letter)
            tags[letter] = tag
            print(f"Created tag: {letter}")
            
        return tags
        
    def upload_images(self, dataset_path, tags):
        """Upload images from dataset folders"""
        print("Uploading images...")
        
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            
            if os.path.isdir(folder_path) and folder_name in tags:
                print(f"Uploading images for class: {folder_name}")
                
                image_list = []
                for image_name in os.listdir(folder_path):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(folder_path, image_name)
                        
                        with open(image_path, "rb") as image_contents:
                            image_list.append(ImageFileCreateEntry(
                                name=image_name,
                                contents=image_contents.read(),
                                tag_ids=[tags[folder_name].id]
                            ))
                
                # Upload in batches of 64 (Azure limit)
                batch_size = 64
                for i in range(0, len(image_list), batch_size):
                    batch = image_list[i:i + batch_size]
                    upload_result = self.trainer.create_images_from_files(
                        self.project.id, 
                        ImageFileCreateBatch(images=batch)
                    )
                    
                    if not upload_result.is_batch_successful:
                        print(f"Image batch upload failed for {folder_name}")
                        for image in upload_result.images:
                            print(f"Image status: {image.status}")
                    else:
                        print(f"Uploaded {len(batch)} images for {folder_name}")
    
    def train_model(self):
        """Train the Custom Vision model"""
        print("Training model...")
        iteration = self.trainer.train_project(self.project.id)
        
        # Wait for training to complete
        while iteration.status != "Completed":
            iteration = self.trainer.get_iteration(self.project.id, iteration.id)
            print(f"Training status: {iteration.status}")
            time.sleep(10)
            
        print("Training completed!")
        return iteration
        
    def publish_model(self, iteration, prediction_resource_id):
        """Publish the trained model"""
        publish_iteration_name = "ASL_Model_v1"
        
        self.trainer.publish_iteration(
            self.project.id,
            iteration.id,
            publish_iteration_name,
            prediction_resource_id
        )
        
        print(f"Model published as: {publish_iteration_name}")
        return publish_iteration_name

# Usage example
if __name__ == "__main__":
    # Replace with your Azure Custom Vision credentials
    TRAINING_KEY = "your_training_key_here"
    ENDPOINT = "your_endpoint_here"
    DATASET_PATH = "path_to_your_ASL_Dataset"
    PREDICTION_RESOURCE_ID = "your_prediction_resource_id"
    
    trainer = ASLModelTrainer(TRAINING_KEY, ENDPOINT)
    
    # Create project and get project ID
    project_id = trainer.create_project()
    
    # Create tags for all classes
    tags = trainer.create_tags()
    
    # Upload training images
    trainer.upload_images(DATASET_PATH, tags)
    
    # Train the model
    iteration = trainer.train_model()
    
    # Publish the model
    model_name = trainer.publish_model(iteration, PREDICTION_RESOURCE_ID)
    
    print(f"Training complete! Project ID: {project_id}")
    print(f"Published model: {model_name}")