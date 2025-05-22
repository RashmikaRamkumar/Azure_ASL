# Azure_ASL - ASL Gesture Recognition & Translation System

A real-time American Sign Language (ASL) gesture recognition system that translates hand gestures to speech using Azure Custom Vision and Azure Speech Services. Trained on Kaggle ASL dataset with 36 classes (A-Z letters and 0-9 numbers).

## 🌟 Features

- **Real-time Gesture Recognition**: Live camera feed processing with gesture detection
- **Text-to-Speech**: Automatic conversion of recognized gestures to spoken words
- **Data Augmentation**: Intelligent image augmentation during training for better model performance
- **Interactive Interface**: User-friendly controls with visual feedback
- **Buffer Management**: Word formation from individual letter/number predictions
- **Pause/Resume**: Control recognition flow during use

## 🏗️ System Architecture

```
Camera Feed → Image Preprocessing → Azure Custom Vision → Gesture Prediction → Speech Synthesis
                                                     ↓
                              Word Buffer ← Confidence Filtering ← Result Processing
```

## 📋 Prerequisites

### Software Requirements
- Python 3.7+
- OpenCV
- PIL (Pillow)
- Azure Cognitive Services SDKs

### Azure Services Required
- **Azure Custom Vision**: For gesture recognition model
- **Azure Speech Services**: For text-to-speech conversion

### Hardware Requirements
- Webcam or external camera
- Microphone and speakers (for audio output)
- Minimum 4GB RAM recommended

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Azure_ASL
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Configuration File**
   Create a `config.py` file with your Azure credentials:
   ```python
   # Azure Custom Vision Training
   TRAINING_KEY = "your_training_key_here"
   ENDPOINT = "your_training_endpoint_here"
   
   # Azure Custom Vision Prediction
   PREDICTION_KEY = "your_prediction_key_here"
   PRED_ENDPOINT = "your_prediction_endpoint_here"
   PROJECT_ID = "your_project_id_here"
   PUBLISHED_MODEL_NAME = "Iteration2"  # Advanced trained model
   
   # Azure Speech Service
   SPEECH_KEY = "your_speech_key_here"
   SPEECH_REGION = "your_region_here"  # e.g., "eastus"
   
   # Dataset Configuration
   DATASET_PATH = "path/to/your/dataset"
   
   # Optional Configuration
   CONFIDENCE_THRESHOLD = 0.7
   CAMERA_INDEX = 0
   FRAME_WIDTH = 640
   FRAME_HEIGHT = 480
   IMAGE_SIZE = (224, 224)
   PREDICTION_DELAY = 1.0
   ```

## 📊 Dataset Structure

This project uses the **Kaggle ASL Dataset** with the following structure:

```
dataset/
├── 0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (~70 images)
├── 1/
│   ├── image1.jpg
│   └── ... (~70 images)
├── 2/
│   └── ... (~70 images)
├── ...
├── 9/
│   └── ... (~70 images)
├── A/
│   ├── image1.jpg
│   └── ... (~70 images)
├── B/
│   └── ... (~70 images)
├── ...
└── Z/
    └── ... (~70 images)
```

**Dataset Details:**
- **36 total classes**: Numbers (0-9) + Letters (A-Z)
- **~70 images per class**: Approximately 2,520 total images
- **Source**: Kaggle ASL Alphabet Dataset
- **Quality**: High-resolution images with clear hand gestures
- **Diversity**: Various backgrounds, lighting conditions, and hand positions

## 🎯 Usage

### Step 1: Training the Model

1. **Prepare your dataset** with the Kaggle ASL dataset structure
2. **Update config.py** with your dataset path and Azure credentials
3. **Run the training script**:
   ```bash
   python aslTraining2.py
   ```

The training process will:
- Create a Custom Vision project
- Set up 36 tags (0-9, A-Z) based on your dataset folders
- Apply data augmentation (rotation, brightness, contrast adjustments)
- Upload augmented images to Azure Custom Vision
- Train the model with advanced parameters
- Display performance metrics
- Create **Iteration2** (advanced trained model)

### Step 2: Running Real-time Translation

After training completes and **Iteration2** is published:

```bash
python asl_translator.py
```

### Controls

- **Q**: Quit the application
- **C**: Clear the word buffer
- **S**: Speak the current buffer contents
- **SPACE**: Pause/Resume gesture recognition

## 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CONFIDENCE_THRESHOLD` | Minimum confidence for gesture acceptance | 0.7 |
| `CAMERA_INDEX` | Camera device index | 0 |
| `FRAME_WIDTH` | Camera frame width | 640 |
| `FRAME_HEIGHT` | Camera frame height | 480 |
| `IMAGE_SIZE` | Image size for model input | (224, 224) |
| `PREDICTION_DELAY` | Delay between predictions (seconds) | 1.0 |

## 🎛️ Model Training Parameters

The training script (`aslTraining2.py`) includes several optimizations:

- **Data Augmentation**: 2x augmentation factor with rotation, brightness, and contrast adjustments
- **Batch Processing**: Images uploaded in batches of 8 to manage API limits
- **Quality Control**: Images validated for size and format
- **Progress Monitoring**: Real-time training status updates
- **Advanced Training**: **Iteration2** uses enhanced training parameters for better accuracy
- **36 Classes**: Complete ASL alphabet (A-Z) and numbers (0-9)
- **Large Dataset**: ~2,520 total images with ~70 images per class

## 📈 Performance Optimization

### For Better Recognition:
1. **Ensure good lighting** when capturing training images and during use
2. **Use consistent hand positioning** within the designated area
3. **Include diverse backgrounds** in training data
4. **Maintain steady hand positions** during recognition
5. **Allow proper delay** between gestures

### Troubleshooting Common Issues:

**Low Confidence Predictions:**
- Add more training images for problematic gestures
- Improve lighting conditions
- Ensure hand is within the green rectangle
- Check camera focus and clarity

**Training Failures:**
- Verify Azure credentials in config.py
- Ensure dataset follows Kaggle ASL structure (0-9, A-Z folders)
- Check internet connectivity
- Verify Azure service quotas
- Ensure each folder has sufficient images (~70 per class)

**Speech Issues:**
- Check Azure Speech Service credentials
- Verify audio output devices
- Ensure proper region configuration

## 📈 Model Performance

The **Iteration2** model provides enhanced recognition for:

- **36 ASL Classes**: Complete alphabet (A-Z) and numbers (0-9)
- **High Accuracy**: Trained on ~2,520 images with data augmentation
- **Real-time Performance**: Optimized for live camera feed processing
- **Robust Recognition**: Works with various lighting and background conditions

The system displays real-time metrics:
- **Current Prediction**: Latest recognized gesture (A-Z, 0-9)
- **Confidence Score**: Model confidence (0-1)
- **Word Buffer**: Accumulated characters for word formation
- **Speaking Status**: Audio output status

## 🔄 Continuous Improvement

To further improve the **Iteration2** model performance:

1. **Collect More Data**: Add images of gestures with poor recognition rates
2. **Diverse Conditions**: Include various lighting, backgrounds, and hand positions
3. **Retrain Regularly**: Create Iteration3 with additional data
4. **Monitor Performance**: Use Azure Custom Vision portal for detailed analytics
5. **Fine-tune Parameters**: Adjust confidence thresholds based on real-world usage

## 🛠️ Development

### Project Structure
```
├── aslTraining2.py           # Model training script (Step 1)
├── asl_translator.py         # Real-time translation app (Step 2)
├── config.py                 # Configuration file
├── requirements.txt          # Python dependencies
├── dataset/                  # Kaggle ASL dataset (0-9, A-Z folders)
└── README.md                 # This file
```

### Dependencies
```
<!-- opencv-python>=4.5.0
Pillow>=8.0.0
azure-cognitiveservices-vision-customvision>=3.1.0
azure-cognitiveservices-speech>=1.19.0
numpy>=1.19.0 -->
```

## 🔒 Security Notes

- Keep Azure credentials secure and never commit them to version control
- Use environment variables for production deployments
- Regularly rotate API keys
- Monitor Azure service usage and costs

<!-- ## 📝 License

[Include your license information here]

## 🤝 Contributing

[Include contribution guidelines here] -->

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify Azure service status
3. Review Azure Custom Vision and Speech Service documentation
4. [Include contact information or issue tracker]

<!-- ## 🙏 Acknowledgments

- Azure Cognitive Services for ML capabilities
- OpenCV community for computer vision tools
- ASL community for gesture standards -->

---

**Note**: This system is designed for educational and assistive purposes. Recognition accuracy may vary based on training data quality, lighting conditions, and individual signing variations.