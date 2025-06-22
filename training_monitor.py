#!/usr/bin/env python3
"""
Doodle Jump AI Training Monitor

This system monitors training progress and provides insights for improvement.
Designed to be safe and avoid system crashes.
"""

import os
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional

class TrainingMonitor:
    """Monitor and analyze AI training progress."""
    
    def __init__(self):
        self.model_info = self._scan_models()
        self.training_data = self._load_training_data()
        
    def _scan_models(self) -> Dict:
        """Scan for existing model files."""
        models = {}
        
        # Check for PyTorch models
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                try:
                    # Extract height from filename if possible
                    if 'height' in file:
                        height = int(file.split('_')[-1].split('.')[0])
                        models[file] = {
                            'height': height,
                            'size_mb': os.path.getsize(file) / (1024 * 1024),
                            'modified': datetime.fromtimestamp(os.path.getmtime(file))
                        }
                    else:
                        models[file] = {
                            'height': 0,
                            'size_mb': os.path.getsize(file) / (1024 * 1024),
                            'modified': datetime.fromtimestamp(os.path.getmtime(file))
                        }
                except:
                    pass
        
        return models
    
    def _load_training_data(self) -> Dict:
        """Load any existing training data."""
        training_files = []
        
        # Check for training logs
        if os.path.exists('data'):
            for file in os.listdir('data'):
                if file.endswith('.json'):
                    training_files.append(f'data/{file}')
        
        # Check for tensorboard logs
        if os.path.exists('tensorboard_logs'):
            for root, dirs, files in os.walk('tensorboard_logs'):
                for file in files:
                    if file.startswith('events.out'):
                        training_files.append(os.path.join(root, file))
        
        return {'files': training_files, 'count': len(training_files)}
    
    def generate_report(self):
        """Generate comprehensive training report."""
        print("🦘 DOODLE JUMP AI TRAINING MONITOR")
        print("=" * 50)
        
        # Model Analysis
        print("📂 MODEL ANALYSIS:")
        if self.model_info:
            best_model = max(self.model_info.items(), key=lambda x: x[1]['height'])
            print(f"   🏆 Best Model: {best_model[0]}")
            print(f"   📈 Best Height: {best_model[1]['height']}")
            print(f"   📊 Total Models: {len(self.model_info)}")
            
            print(f"\n   📋 All Models:")
            for model, info in sorted(self.model_info.items(), key=lambda x: x[1]['height'], reverse=True):
                print(f"      • {model}: Height {info['height']}, {info['size_mb']:.1f}MB")
        else:
            print("   ❌ No trained models found")
        
        # Training Data Analysis
        print(f"\n📊 TRAINING DATA:")
        print(f"   📁 Training files found: {self.training_data['count']}")
        
        if self.training_data['files']:
            print("   📄 Files:")
            for file in self.training_data['files'][:5]:  # Show first 5
                print(f"      • {file}")
            if len(self.training_data['files']) > 5:
                print(f"      ... and {len(self.training_data['files']) - 5} more")
        
        # System Status
        print(f"\n🖥️  SYSTEM STATUS:")
        self._check_system_status()
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        self._generate_recommendations()
    
    def _check_system_status(self):
        """Check system requirements and status."""
        try:
            import torch
            print(f"   ✅ PyTorch: {torch.__version__}")
        except:
            print("   ❌ PyTorch: Not available")
        
        try:
            import gymnasium as gym
            print("   ✅ Gymnasium: Available")
        except:
            print("   ❌ Gymnasium: Not available")
        
        try:
            import stable_baselines3
            print("   ✅ Stable Baselines3: Available")
        except:
            print("   ❌ Stable Baselines3: Not available")
        
        # Check directories
        directories = ['src', 'models', 'data', 'tensorboard_logs']
        for directory in directories:
            if os.path.exists(directory):
                print(f"   ✅ {directory}/: Present")
            else:
                print(f"   ❌ {directory}/: Missing")
                try:
                    os.makedirs(directory, exist_ok=True)
                    print(f"      ➡️ Created {directory}/")
                except:
                    pass
    
    def _generate_recommendations(self):
        """Generate training recommendations based on current state."""
        recommendations = []
        
        if not self.model_info:
            recommendations.append("🔄 Start fresh training - no existing models found")
        else:
            best_height = max(info['height'] for info in self.model_info.values())
            if best_height < 500:
                recommendations.append("📈 Focus on basic jumping mechanics")
                recommendations.append("🎯 Target: Reach height 500")
            elif best_height < 1000:
                recommendations.append("🚀 Good progress! Focus on advanced techniques")
                recommendations.append("🎯 Target: Reach height 1000")
            else:
                recommendations.append("🏆 Excellent! Focus on consistency and optimization")
                recommendations.append("🎯 Target: Improve average performance")
        
        if self.training_data['count'] == 0:
            recommendations.append("📊 Set up training data logging")
            recommendations.append("📈 Enable tensorboard monitoring")
        
        # Safe training approach recommendations
        recommendations.extend([
            "🛡️ Use safe training mode to avoid system crashes",
            "💾 Save models frequently during training",
            "📱 Monitor system resources during training",
            "🔍 Test components individually before full training"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def plot_progress(self):
        """Plot training progress if data is available."""
        if not self.model_info:
            print("📊 No model data available for plotting")
            return
        
        # Create progress plot
        models = sorted(self.model_info.items(), key=lambda x: x[1]['modified'])
        heights = [info['height'] for _, info in models]
        dates = [info['modified'] for _, info in models]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, heights, 'bo-', linewidth=2, markersize=8)
        plt.title('Doodle Jump AI Training Progress')
        plt.xlabel('Training Date')
        plt.ylabel('Max Height Achieved')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add annotations
        for i, (date, height) in enumerate(zip(dates, heights)):
            if height > 0:
                plt.annotate(f'{height}', (date, height), 
                           textcoords="offset points",
                           xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig('training_progress_monitor.png', dpi=300)
        plt.show()
        print("📈 Progress plot saved as 'training_progress_monitor.png'")
    
    def create_safe_training_plan(self):
        """Create a safe training plan based on current state."""
        plan_file = "safe_training_plan.md"
        
        with open(plan_file, 'w') as f:
            f.write("# Safe Doodle Jump AI Training Plan\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Current Status\n")
            if self.model_info:
                best_height = max(info['height'] for info in self.model_info.values())
                f.write(f"- Best Height Achieved: {best_height}\n")
                f.write(f"- Total Models: {len(self.model_info)}\n")
            else:
                f.write("- No existing models found\n")
            
            f.write(f"- Training Data Files: {self.training_data['count']}\n\n")
            
            f.write("## Safe Training Steps\n")
            f.write("1. **Environment Check**: Ensure all dependencies are working\n")
            f.write("2. **Component Testing**: Test each component individually\n")
            f.write("3. **Small Batch Training**: Start with 10-25 episodes\n")
            f.write("4. **Progress Monitoring**: Check progress after each batch\n")
            f.write("5. **Model Saving**: Save models after each successful batch\n")
            f.write("6. **Gradual Scaling**: Increase batch size as stability improves\n\n")
            
            f.write("## Troubleshooting\n")
            f.write("- If bus errors occur: Reduce batch size or memory usage\n")
            f.write("- If training stalls: Check game detection and input systems\n")
            f.write("- If models don't improve: Adjust hyperparameters\n\n")
            
            f.write("## Next Steps\n")
            if self.model_info:
                f.write("- Continue training from best existing model\n")
                f.write("- Focus on consistency improvements\n")
            else:
                f.write("- Start with basic training setup\n")
                f.write("- Establish baseline performance\n")
        
        print(f"📋 Safe training plan saved to '{plan_file}'")

def main():
    """Main function to run the training monitor."""
    monitor = TrainingMonitor()
    
    print("🔍 Analyzing training system...")
    monitor.generate_report()
    
    print("\n📋 Creating safe training plan...")
    monitor.create_safe_training_plan()
    
    # Try to plot progress
    try:
        print("\n📊 Generating progress plot...")
        monitor.plot_progress()
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")
    
    print("\n✅ Training monitoring complete!")

if __name__ == "__main__":
    main() 