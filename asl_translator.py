import cv2
import numpy as np
import time
import threading
from io import BytesIO
from PIL import Image
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import config

class ASLTranslator:
    def __init__(self):
        print("🤖 Initializing ASL Translator...")
        
        # Validate configuration
        self.validate_config()
        
        # Custom Vision setup
        print("Setting up Custom Vision...")
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": config.PREDICTION_KEY})
        self.predictor = CustomVisionPredictionClient(config.ENDPOINT, prediction_credentials)
        
        # Speech setup
        print("Setting up Speech Service...")
        self.speech_config = speechsdk.SpeechConfig(
            subscription=config.SPEECH_KEY, 
            region=config.SPEECH_REGION
        )
        self.speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        
        # Translation variables
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.last_spoken = ""
        self.speak_threshold = config.CONFIDENCE_THRESHOLD
        self.word_buffer = []
        self.last_prediction_time = time.time()
        self.is_speaking = False
        
        # Camera setup
        print("Setting up camera...")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise Exception("❌ Could not open camera")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        print("✓ ASL Translator initialized successfully!")
        
    def validate_config(self):
        """Validate configuration settings"""
        required_fields = [
            ('PREDICTION_KEY', config.PREDICTION_KEY),
            ('ENDPOINT', config.ENDPOINT),
            ('PROJECT_ID', config.PROJECT_ID),
            ('SPEECH_KEY', config.SPEECH_KEY),
            ('SPEECH_REGION', config.SPEECH_REGION)
        ]
        
        for field_name, field_value in required_fields:
            if not field_value or field_value.startswith("YOUR_"):
                raise Exception(f"❌ Please set {field_name} in config.py")
        
    def preprocess_image(self, frame):
        """Preprocess the frame for better prediction"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            resized = cv2.resize(rgb_frame, config.IMAGE_SIZE)
            
            # Apply some preprocessing for better recognition
            # Normalize brightness
            lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            resized = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(resized)
            
            # Convert to bytes
            image_buffer = BytesIO()
            pil_image.save(image_buffer, format='JPEG', quality=95)
            image_buffer.seek(0)
            
            return image_buffer
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
        
    def predict_gesture(self, image_buffer):
        """Predict ASL gesture using Custom Vision"""
        try:
            results = self.predictor.classify_image(
                config.PROJECT_ID, 
                config.PUBLISHED_MODEL_NAME, 
                image_buffer.getvalue()
            )
            
            if results.predictions:
                # Get the prediction with highest probability
                top_prediction = max(results.predictions, key=lambda x: x.probability)
                return top_prediction.tag_name, top_prediction.probability
            
        except Exception as e:
            if "NotFound" in str(e):
                print("❌ Model not found. Make sure you've trained and published the model.")
            else:
                print(f"Prediction error: {e}")
            
        return None, 0.0
        
    def speak_text(self, text):
        """Convert text to speech using Azure Speech Service"""
        if self.is_speaking:
            return
            
        self.is_speaking = True
        
        try:
            # Create SSML for better speech quality
            ssml = f"""
            <speak version='1.0' xml:lang='en-US'>
                <voice name='en-US-AriaNeural'>
                    <prosody rate='medium' pitch='medium' volume='loud'>
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            result = self.speech_synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"🔊 Spoke: {text}")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print(f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.error_details:
                    print(f"Error details: {cancellation_details.error_details}")
                
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.is_speaking = False
            
    def process_prediction(self, prediction, confidence):
        """Process prediction and manage speech output"""
        current_time = time.time()
        
        if confidence > self.speak_threshold:
            self.current_prediction = prediction
            self.prediction_confidence = confidence
            
            # Add to word buffer if it's a new prediction
            if (prediction != self.last_spoken and 
                current_time - self.last_prediction_time > config.PREDICTION_DELAY):
                
                self.word_buffer.append(prediction)
                self.last_prediction_time = current_time
                
                # Speak individual letters/numbers immediately
                threading.Thread(target=self.speak_text, args=(prediction,), daemon=True).start()
                self.last_spoken = prediction
                
                # Auto-speak words when buffer reaches certain length
                if len(self.word_buffer) >= 5:
                    word = ''.join(self.word_buffer)
                    threading.Thread(target=self.speak_text, args=(f"Word: {word}",), daemon=True).start()
                    self.word_buffer = []
                    
    def draw_interface(self, frame):
        """Draw the user interface on the frame"""
        height, width = frame.shape[:2]
        
        # Draw main info panel
        cv2.rectangle(frame, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 150), (255, 255, 255), 2)
        
        # Current prediction info
        pred_text = f"Prediction: {self.current_prediction}"
        conf_text = f"Confidence: {self.prediction_confidence:.2f}"
        buffer_text = f"Buffer: {''.join(self.word_buffer)}"
        status_text = f"Speaking: {'Yes' if self.is_speaking else 'No'}"
        
        cv2.putText(frame, pred_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, buffer_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions panel
        instructions = [
            "Controls:",
            "Q - Quit",
            "C - Clear buffer", 
            "S - Speak buffer",
            "SPACE - Pause/Resume"
        ]
        
        y_start = 170
        cv2.rectangle(frame, (10, y_start), (300, y_start + len(instructions) * 25 + 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, y_start), (300, y_start + len(instructions) * 25 + 20), (255, 255, 255), 2)
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, y_start + 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        # Hand region indicator
        hand_x1, hand_y1 = width//4, height//4
        hand_x2, hand_y2 = 3*width//4, 3*height//4
        
        cv2.rectangle(frame, (hand_x1, hand_y1), (hand_x2, hand_y2), (0, 255, 0), 3)
        cv2.putText(frame, "Place hand here", (hand_x1, hand_y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame, (hand_x1, hand_y1, hand_x2, hand_y2)
        
    def run(self):
        """Main loop for real-time ASL translation"""
        print("\n🚀 Starting ASL Gesture to Speech Translator...")
        print("=" * 50)
        print("Controls:")
        print("  Q - Quit")
        print("  C - Clear word buffer")
        print("  S - Speak current buffer")
        print("  SPACE - Pause/Resume recognition")
        print("=" * 50)
        
        