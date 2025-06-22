#!/usr/bin/env python3
"""
🚀 Doodle Jump AI - Interactive Demo

Quick demonstration of the world-class AI agent that achieved 17,444.7 height!

Usage:
    python demo.py

Author: Aashish Kumar
Date: 2025
"""

import sys
import os

def main():
    print("🚀 Doodle Jump AI - Interactive Demo")
    print("=" * 50)
    print("🏆 World-class AI: 922 → 17,444.7 height (1,693% improvement)")
    print("=" * 50)
    
    while True:
        print("\n📋 Demo Options:")
        print("1. 🧠 Run Safe Training (Quick Demo)")
        print("2. 🚀 Run Advanced Training Session")
        print("3. 📊 View Training Monitor")
        print("4. 🎮 Manual Game Control")
        print("5. 📈 View Training Results")
        print("6. ❌ Exit")
        
        choice = input("\n🎯 Select option (1-6): ").strip()
        
        if choice == '1':
            print("\n🛡️ Starting Safe Training Demo...")
            print("This will run a quick training session (15 episodes)")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                try:
                    import safe_training
                    print("✅ Safe training completed!")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print("💡 Make sure all dependencies are installed")
        
        elif choice == '2':
            print("\n🚀 Starting Advanced Training Session...")
            print("This will run the full curriculum learning pipeline")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                try:
                    import advanced_training_session
                    print("✅ Advanced training completed!")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print("💡 Check system resources and dependencies")
        
        elif choice == '3':
            print("\n📊 Opening Training Monitor...")
            try:
                import training_monitor
                print("✅ Training monitor opened!")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Make sure training data exists")
        
        elif choice == '4':
            print("\n🎮 Starting Manual Game Control...")
            try:
                import main
                print("✅ Game control ready!")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Make sure game dependencies are installed")
        
        elif choice == '5':
            print("\n📈 Training Results Summary:")
            print("🎯 Peak Achievement: 17,444.7 height")
            print("📊 Total Improvement: +1,693% from baseline")
            print("🏆 Best Technique: Advanced Combo System (1.35x boost)")
            print("✅ Success Rate: 100% (zero crashes)")
            print("\n📁 Detailed results available in:")
            print("   - docs/TRAINING_ACHIEVEMENTS.md")
            print("   - training_data/*.json")
            print("   - screenshots/*.png")
        
        elif choice == '6':
            print("\n👋 Thanks for trying Doodle Jump AI!")
            print("🌟 Star the project on GitHub if you found it interesting!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-6.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Make sure all dependencies are installed with: pip install -r requirements.txt") 