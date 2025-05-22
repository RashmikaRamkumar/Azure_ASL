import os
import cv2
import numpy as np
from PIL import Image
import config
from collections import defaultdict
import matplotlib.pyplot as plt

class DatasetQualityAnalyzer:
    def __init__(self):
        self.dataset_path = config.DATASET_PATH
        self.quality_issues = defaultdict(list)
        
    def analyze_image_quality(self):
        """Analyze image quality metrics that affect model performance"""
        print("🔍 ANALYZING IMAGE QUALITY FOR ML PERFORMANCE")
        print("=" * 60)
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return
        
        class_stats = {}
        overall_stats = {
            'brightness': [],
            'contrast': [],
            'sharpness': [],
            'sizes': [],
            'file_sizes': []
        }
        
        folders = [f for f in os.listdir(self.dataset_path) 
                  if os.path.isdir(os.path.join(self.dataset_path, f))]
        folders.sort()
        
        for folder_name in folders:
            folder_path = os.path.join(self.dataset_path, folder_name)
            print(f"\n📂 Analyzing class: {folder_name}")
            
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not image_files:
                continue
                
            class_metrics = {
                'brightness': [],
                'contrast': [],
                'sharpness': [],
                'variety_score': 0
            }
            
            # Analyze sample of images
            sample_size = min(10, len(image_files))
            sample_images = image_files[:sample_size]
            
            for img_name in sample_images:
                img_path = os.path.join(folder_path, img_name)
                metrics = self._analyze_single_image(img_path)
                
                if metrics:
                    class_metrics['brightness'].append(metrics['brightness'])
                    class_metrics['contrast'].append(metrics['contrast'])
                    class_metrics['sharpness'].append(metrics['sharpness'])
                    
                    overall_stats['brightness'].append(metrics['brightness'])
                    overall_stats['contrast'].append(metrics['contrast'])
                    overall_stats['sharpness'].append(metrics['sharpness'])
                    overall_stats['sizes'].append(metrics['size'])
                    overall_stats['file_sizes'].append(metrics['file_size'])
            
            # Calculate class statistics
            if class_metrics['brightness']:
                avg_brightness = np.mean(class_metrics['brightness'])
                avg_contrast = np.mean(class_metrics['contrast'])
                avg_sharpness = np.mean(class_metrics['sharpness'])
                
                brightness_var = np.var(class_metrics['brightness'])
                contrast_var = np.var(class_metrics['contrast'])
                
                class_stats[folder_name] = {
                    'count': len(image_files),
                    'avg_brightness': avg_brightness,
                    'avg_contrast': avg_contrast,
                    'avg_sharpness': avg_sharpness,
                    'brightness_variety': brightness_var,
                    'contrast_variety': contrast_var
                }
                
                # Check for quality issues
                self._check_quality_issues(folder_name, class_metrics, len(image_files))
                
                print(f"  ✓ {len(image_files)} images analyzed")
                print(f"    Brightness: {avg_brightness:.2f} (variety: {brightness_var:.2f})")
                print(f"    Contrast: {avg_contrast:.2f} (variety: {contrast_var:.2f})")
                print(f"    Sharpness: {avg_sharpness:.2f}")
        
        self._print_quality_report(class_stats, overall_stats)
        self._provide_recommendations(class_stats)
        
    def _analyze_single_image(self, img_path):
        """Analyze quality metrics for a single image"""
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return None
                
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Sharpness (using Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # File info
            file_size = os.path.getsize(img_path)
            height, width = img.shape[:2]
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'size': (width, height),
                'file_size': file_size
            }
            
        except Exception as e:
            print(f"    ❌ Error analyzing {img_path}: {e}")
            return None
    
    def _check_quality_issues(self, class_name, metrics, total_count):
        """Identify potential quality issues"""
        if total_count < 15:
            self.quality_issues[class_name].append(f"Low sample count: {total_count} (recommend 30+)")
        
        if metrics['brightness']:
            avg_brightness = np.mean(metrics['brightness'])
            brightness_var = np.var(metrics['brightness'])
            
            if avg_brightness < 80:
                self.quality_issues[class_name].append("Images too dark (avg brightness < 80)")
            
            if avg_brightness > 200:
                self.quality_issues[class_name].append("Images too bright (avg brightness > 200)")
            
            if brightness_var < 100:
                self.quality_issues[class_name].append("Low lighting variety (may cause overfitting)")
        
        if metrics['contrast']:
            avg_contrast = np.mean(metrics['contrast'])
            if avg_contrast < 30:
                self.quality_issues[class_name].append("Low contrast images (avg < 30)")
        
        if metrics['sharpness']:
            avg_sharpness = np.mean(metrics['sharpness'])
            if avg_sharpness < 100:
                self.quality_issues[class_name].append("Blurry images detected (sharpness < 100)")
    
    def _print_quality_report(self, class_stats, overall_stats):
        """Print comprehensive quality report"""
        print(f"\n{'='*60}")
        print("📊 QUALITY ANALYSIS REPORT")
        print(f"{'='*60}")
        
        # Overall statistics
        if overall_stats['brightness']:
            print(f"\n🌍 Overall Dataset Statistics:")
            print(f"  Average brightness: {np.mean(overall_stats['brightness']):.2f}")
            print(f"  Average contrast: {np.mean(overall_stats['contrast']):.2f}")
            print(f"  Average sharpness: {np.mean(overall_stats['sharpness']):.2f}")
            print(f"  Average file size: {np.mean(overall_stats['file_sizes'])/1024:.1f} KB")
        
        # Class-by-class issues
        print(f"\n⚠️ Quality Issues by Class:")
        has_issues = False
        for class_name, issues in self.quality_issues.items():
            if issues:
                has_issues = True
                print(f"  {class_name}:")
                for issue in issues:
                    print(f"    - {issue}")
        
        if not has_issues:
            print("  ✅ No major quality issues detected!")
        
        # Classes with best/worst metrics
        if class_stats:
            best_contrast = max(class_stats.items(), key=lambda x: x[1]['avg_contrast'])
            worst_contrast = min(class_stats.items(), key=lambda x: x[1]['avg_contrast'])
            
            print(f"\n📈 Quality Rankings:")
            print(f"  Best contrast: {best_contrast[0]} ({best_contrast[1]['avg_contrast']:.2f})")
            print(f"  Worst contrast: {worst_contrast[0]} ({worst_contrast[1]['avg_contrast']:.2f})")
    
    def _provide_recommendations(self, class_stats):
        """Provide specific recommendations for improvement"""
        print(f"\n🎯 RECOMMENDATIONS FOR BETTER MODEL PERFORMANCE")
        print(f"{'='*60}")
        
        # Data quantity recommendations
        low_count_classes = [name for name, stats in class_stats.items() if stats['count'] < 30]
        if low_count_classes:
            print(f"\n📸 Increase Data Volume:")
            print(f"  Classes needing more images: {', '.join(low_count_classes[:5])}")
            print(f"  Recommendation: Aim for 50+ images per class")
        
        # Data quality recommendations
        print(f"\n🎨 Improve Data Diversity:")
        print(f"  1. Vary lighting conditions (bright, dim, natural, artificial)")
        print(f"  2. Change backgrounds (plain, textured, different colors)")
        print(f"  3. Multiple hand positions and angles")
        print(f"  4. Different distances from camera")
        print(f"  5. Include both left and right hands")
        print(f"  6. Various skin tones and hand sizes")
        
        # Technical recommendations
        print(f"\n⚙️ Technical Improvements:")
        print(f"  1. Ensure good image sharpness (avoid blurry photos)")
        print(f"  2. Maintain consistent image size (400x400px is good)")
        print(f"  3. Use good contrast between hand and background")
        print(f"  4. Avoid overexposed or underexposed images")
        
        # Model training recommendations
        print(f"\n🚀 Training Strategy:")
        print(f"  1. Use data augmentation (rotation, brightness, contrast)")
        print(f"  2. Consider transfer learning if available")
        print(f"  3. Monitor per-class performance and focus on weak classes")
        print(f"  4. Use proper train/validation split")
        
        # Specific Azure Custom Vision tips
        print(f"\n☁️ Azure Custom Vision Specific:")
        print(f"  1. Minimum 15 images per tag (30+ recommended)")
        print(f"  2. Maximum 10,000 images per project")
        print(f"  3. Use 'General (compact)' domain for mobile deployment")
        print(f"  4. Test with 'Quick Test' feature before publishing")

def main():
    analyzer = DatasetQualityAnalyzer()
    analyzer.analyze_image_quality()

if __name__ == "__main__":
    main()