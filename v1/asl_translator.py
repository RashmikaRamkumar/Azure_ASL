import cv2
import numpy as np
import time
import threading
from io import BytesIO
from PIL import Image
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

class ASLTranslator:
    def __init__(self, prediction_key, prediction_endpoint, project_id, 
                 published_name, speech_key, speech_region):
        
        # Custom Vision setup
        self.prediction_key = prediction_key
        self.prediction_endpoint = prediction_endpoint
        self.project_id = project_id
        self.published_name = published_name
        
        # Initialize Custom Vision predictor
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        self.predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)
        
        # Speech setup
        self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        self.speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        
        # Translation variables
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.last_spoken = ""
        self.speak_threshold = 0.7  # Minimum confidence to speak
        self.word_buffer = []
        self.last_prediction_time = time.time()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def preprocess_image(self, frame):
        """Preprocess the frame for better prediction"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        resized = cv2.resize(rgb_frame, (224, 224))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized)
        
        # Convert to bytes
        image_buffer = BytesIO()
        pil_image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)
        
        return image_buffer
        
    def predict_gesture(self, image_buffer):
        """Predict ASL gesture using Custom Vision"""
        try:
            results = self.predictor.classify_image(
                self.project_id, 
                self.published_name, 
                image_buffer.getvalue()
            )
            
            if results.predictions:
                # Get the prediction with highest probability
                top_prediction = max(results.predictions, key=lambda x: x.probability)
                return top_prediction.tag_name, top_prediction.probability
            
        except Exception as e:
            print(f"Prediction error: {e}")
            
        return None, 0.0
        
    def speak_text(self, text):
        """Convert text to speech using Azure Speech Service"""
        try:
            # Create SSML for better speech quality
            ssml = f"""
            <speak version='1.0' xml:lang='en-US'>
                <voice name='en-US-AriaNeural'>
                    <prosody rate='medium' pitch='medium'>
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            result = self.speech_synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"Spoke: {text}")
            else:
                print(f"Speech synthesis failed: {result.reason}")
                
        except Exception as e:
            print(f"Speech error: {e}")
            
    def process_prediction(self, prediction, confidence):
        """Process prediction and manage speech output"""
        current_time = time.time()
        
        if confidence > self.speak_threshold:
            self.current_prediction = prediction
            self.prediction_confidence = confidence
            
            # Add to word buffer if it's a new prediction
            if (prediction != self.last_spoken and 
                current_time - self.last_prediction_time > 1.0):  # 1 second delay
                
                self.word_buffer.append(prediction)
                self.last_prediction_time = current_time
                
                # Speak individual letters/numbers immediately
                threading.Thread(target=self.speak_text, args=(prediction,)).start()
                self.last_spoken = prediction
                
                # If buffer has multiple characters, form words
                if len(self.word_buffer) >= 3:
                    word = ''.join(self.word_buffer)
                    threading.Thread(target=self.speak_text, args=(f"Word: {word}",)).start()
                    self.word_buffer = []
                    
    def draw_interface(self, frame):
        """Draw the user interface on the frame"""
        # Draw prediction info
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Current prediction
        pred_text = f"Prediction: {self.current_prediction}"
        conf_text = f"Confidence: {self.prediction_confidence:.2f}"
        buffer_text = f"Buffer: {''.join(self.word_buffer)}"
        
        cv2.putText(frame, pred_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, conf_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, buffer_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Instructions:",
            "- Show ASL gestures to camera",
            "- 'q' to quit",
            "- 'c' to clear buffer",
            "- 's' to speak current buffer"
        ]
        
        y_pos = 150
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            
        # Draw region of interest for hand detection
        cv2.rectangle(frame, (200, 100), (500, 400), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand here", (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
        
    def run(self):
        """Main loop for real-time ASL translation"""
        print("Starting ASL Gesture to Speech Translator...")
        print("Press 'q' to quit, 'c' to clear buffer, 's' to speak buffer")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract hand region (you can adjust these coordinates)
            hand_region = frame[100:400, 200:500]
            
            # Preprocess the hand region for prediction
            processed_image = self.preprocess_image(hand_region)
            
            # Get prediction
            prediction, confidence = self.predict_gesture(processed_image)
            
            if prediction:
                self.process_prediction(prediction, confidence)
            
            # Draw interface
            frame = self.draw_interface(frame)
            
            # Show frame
            cv2.imshow('ASL Gesture to Speech Translator', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.word_buffer = []
                print("Buffer cleared")
            elif key == ord('s'):
                if self.word_buffer:
                    word = ''.join(self.word_buffer)
                    threading.Thread(target=self.speak_text, args=(word,)).start()
                    print(f"Speaking buffer: {word}")
                    
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("ASL Translator stopped.")

# Usage
if __name__ == "__main__":
    # Replace with your Azure credentials
    PREDICTION_KEY = "your_prediction_key"
    PREDICTION_ENDPOINT = "your_prediction_endpoint"
    PROJECT_ID = "your_project_id"
    PUBLISHED_NAME = "ASL_Model_v1"
    SPEECH_KEY = "your_speech_key"
    SPEECH_REGION = "your_speech_region"
    
    translator = ASLTranslator(
        prediction_key=PREDICTION_KEY,
        prediction_endpoint=PREDICTION_ENDPOINT,
        project_id=PROJECT_ID,
        published_name=PUBLISHED_NAME,
        speech_key=SPEECH_KEY,
        speech_region=SPEECH_REGION
    )
    
    translator.run()