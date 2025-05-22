# config.py
# Azure Custom Vision Configuration
TRAINING_KEY = "your_custom_vision_training_key"
PREDICTION_KEY = "your_custom_vision_prediction_key"
ENDPOINT = "https://your-resource-name.cognitiveservices.azure.com/"
PREDICTION_RESOURCE_ID = "/subscriptions/your-subscription-id/resourceGroups/your-resource-group/providers/Microsoft.CognitiveServices/accounts/your-prediction-resource-name"

# Azure Speech Service Configuration  
SPEECH_KEY = "your_speech_service_key"
SPEECH_REGION = "your_speech_service_region"  # e.g., "eastus"

# Project Configuration
PROJECT_NAME = "ASL_Gesture_Recognition"
PUBLISHED_MODEL_NAME = "ASL_Model_v1"

# Dataset Configuration
DATASET_PATH = "path/to/your/ASL_Dataset"

# Model Parameters
CONFIDENCE_THRESHOLD = 0.7
PREDICTION_DELAY = 1.0  # seconds between predictions
IMAGE_SIZE = (224, 224)

# Camera Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Hand Region Coordinates (adjust based on your camera setup)
HAND_REGION = {
    'x1': 200,
    'y1': 100, 
    'x2': 500,
    'y2': 400
}