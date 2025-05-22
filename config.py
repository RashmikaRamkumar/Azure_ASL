# config.py
# Replace these with your actual Azure credentials

# Azure Custom Vision Configuration
TRAINING_KEY = "EZ3qwQ12e0zuuuu6U0P7p0dRdiPLKjKrfLQSOXsk90KDHsmAMRucJQQJ99BEACYeBjFXJ3w3AAAJACOGPV1X"  # From asl-customvision-training resource
PREDICTION_KEY = "CniUcwEAI6sqDtHfLwtjmY9XRvZ3hc4kBQIOaXk15XUryu5Rtqf7JQQJ99BEACYeBjFXJ3w3AAAIACOGz48A"  # From asl-customvision-prediction resource
ENDPOINT = "https://asltrain.cognitiveservices.azure.com/"  # From training resource (e.g., https://eastus.api.cognitive.microsoft.com/)
PREDICTION_RESOURCE_ID = "/subscriptions/dbc68e33-eadd-4766-9e31-f0a1ca8f7ec3/resourceGroups/resource1/providers/Microsoft.CognitiveServices/accounts/asl-customvision-prediction"  # Full resource ID from prediction resource properties

# Azure Speech Service Configuration  
SPEECH_KEY = "Ae4IRMaxROuc3n5q8tMeliYsMyTulDeeprDJq3pcFQU8AanOMhAcJQQJ99BEACYeBjFXJ3w3AAAYACOGqFuh"  # From asl-speech-service resource
SPEECH_REGION = "eastus"  # e.g., "eastus", "westus2"

# Project Configuration (will be filled after training)
PROJECT_ID = "700de036-0ebe-4484-94ce-028f7b6f9e6b"  # Will be generated during training
PUBLISHED_MODEL_NAME = "ASL_Model_v1"

# Dataset Configuration
DATASET_PATH = r"D:\Azure_project\asl_dataset"  # Update this path

CONFIDENCE_THRESHOLD = 0.7
PREDICTION_DELAY = 1.0  # seconds between predictions
IMAGE_SIZE = (224, 224)

# Camera Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


#==================================================
# ✓ Project ID: 700de036-0ebe-4484-94ce-028f7b6f9e6b
# ✓ Iteration ID: 44ba103d-25b8-425e-85eb-788f25acb153
#==================================================

# Example of what your keys should look like:
# TRAINING_KEY = "a1b2c3d4e5f67890123456789abcdef1"
# PREDICTION_KEY = "z9y8x7w6v5u43210987654321fedcba9" 
# SPEECH_KEY = "m1n2o3p4q5r67890123456789hijklmn2"
# ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
# PREDICTION_RESOURCE_ID = "/subscriptions/12345678-abcd-1234-efgh-123456789abc/resourceGroups/asl-resources/providers/Microsoft.CognitiveServices/accounts/asl-customvision-prediction"