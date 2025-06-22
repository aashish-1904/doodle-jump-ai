#!/usr/bin/env python3
"""
Doodle Jump AI Agent - Main Application

This AI agent can play Doodle Jump autonomously, learn optimal jumping strategies,
and track progress including high scores and achievements.

Author: AI Assistant
Date: 2025
"""

import argparse
import time
import sys
import os
import signal
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.game_vision import DoodleVision
from src.game_controller import DoodleController
from src.rl_agent import DoodleJumpAgent
from src.progress_tracker import DoodleProgressTracker

class DoodleJumpAI:
    """Main Doodle Jump AI application."""
    
    def __init__(self):
        print("🦘 Initializing Doodle Jump AI Agent...")
        
        # Initialize components
        self.game_vision = DoodleVision()
        self.game_controller = DoodleController()
        self.progress_tracker = DoodleProgressTracker()
        self.agent = DoodleJumpAgent(self.game_vision, self.game_controller)
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
        
        print("✅ Doodle Jump AI Agent initialized successfully!")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Shutdown signal received. Cleaning up...")
        self.running = False
        self.game_controller.emergency_stop()
        self.progress_tracker.save_progress()
        print("🔄 Cleanup complete. Goodbye!")
        sys.exit(0)
        
    def train_agent(self, episodes: int = 1000):
        """Train the AI agent for specified number of episodes."""
        print(f"🎯 Starting training for {episodes} episodes...")
        
        try:
            # Check if the game is running
            if not self.check_game_status():
                print("❌ Doodle Jump game not detected. Please start the game first.")
                return
                
            print("🎮 Game detected! Starting training...")
            
            start_time = time.time()
            
            # Train the agent
            for episode in range(episodes):
                if not self.running:
                    break
                    
                print(f"\n🔄 Training Episode {episode + 1}/{episodes}")
                
                episode_start_time = time.time()
                
                # Reset environment and get initial state
                obs = self.agent.env.reset()[0]
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done and self.running:
                    # Get action from agent
                    action, _ = self.agent.model.predict(obs)
                    
                    # Execute action and get new state
                    obs, reward, done, _, info = self.agent.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)
                    
                episode_time = time.time() - episode_start_time
                
                # Update progress tracking
                episode_stats = info.get('episode_stats', {})
                self.progress_tracker.update_game_stats(episode_stats)
                self.progress_tracker.update_training_stats(episode_reward, episode_length, episode_time)
                
                # Print episode summary
                max_height = episode_stats.get('max_height', 0)
                print(f"📊 Episode {episode + 1} Results:")
                print(f"   Height: {max_height}")
                print(f"   Reward: {episode_reward:.2f}")
                print(f"   Steps: {episode_length}")
                print(f"   Duration: {episode_time:.1f}s")
                
                # Save progress periodically
                if (episode + 1) % 25 == 0:
                    self.progress_tracker.save_progress()
                    self.agent.save_model(f"models/checkpoint_episode_{episode + 1}")
                    
                # Show progress report periodically
                if (episode + 1) % 100 == 0:
                    self.progress_tracker.print_progress_report()
                    
            total_time = time.time() - start_time
            print(f"🏁 Training completed in {total_time/60:.1f} minutes!")
                    
        except Exception as e:
            print(f"❌ Training error: {e}")
            self.game_controller.emergency_stop()
            
        finally:
            print("💾 Saving final progress...")
            self.progress_tracker.save_progress()
            self.agent.save_model("models/final_model")
            
    def play_game(self, episodes: int = 1, show_debug: bool = True):
        """Play the game using the trained agent."""
        print(f"🎮 Playing {episodes} episode(s) using trained agent...")
        
        try:
            # Check if the game is running
            if not self.check_game_status():
                print("❌ Doodle Jump game not detected. Please start the game first.")
                return
                
            # Load best model if available
            if os.path.exists("models/best_model.zip"):
                self.agent.load_model("models/best_model")
                print("✅ Loaded best trained model")
            else:
                print("⚠️  No trained model found. Using randomly initialized agent.")
                
            # Play episodes
            self.agent.play(num_episodes=episodes, render_debug=show_debug)
            
        except Exception as e:
            print(f"❌ Gameplay error: {e}")
            self.game_controller.emergency_stop()
            
    def evaluate_agent(self, episodes: int = 10):
        """Evaluate the current agent performance."""
        print(f"📊 Evaluating agent performance over {episodes} episodes...")
        
        try:
            if not self.check_game_status():
                print("❌ Doodle Jump game not detected. Please start the game first.")
                return
                
            # Load best model if available
            if os.path.exists("models/best_model.zip"):
                self.agent.load_model("models/best_model")
                
            # Evaluate
            avg_height, avg_reward = self.agent.evaluate(num_episodes=episodes)
            print(f"📈 Results over {episodes} episodes:")
            print(f"   Average Height: {avg_height:.1f}")
            print(f"   Average Reward: {avg_reward:.2f}")
            
            # Show detailed progress report
            self.progress_tracker.print_progress_report()
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            
    def check_game_status(self) -> bool:
        """Check if Doodle Jump game is running and accessible."""
        try:
            # Try to capture a frame
            game_state = self.game_vision.get_game_state()
            frame = game_state['frame']
            
            # Simple check: if frame is not all black/empty
            if frame is not None and frame.mean() > 10:
                print(f"✅ Game detected - State: {game_state.get('game_state', 'unknown')}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"⚠️  Game status check failed: {e}")
            return False
            
    def show_progress(self):
        """Display current progress and statistics."""
        print("📊 Current Progress:")
        self.progress_tracker.print_progress_report()
        
    def export_progress(self, filename: str = None):
        """Export progress data to file."""
        self.progress_tracker.export_stats(filename)
        
    def test_vision(self):
        """Test the computer vision system."""
        print("👁️  Testing computer vision system...")
        
        try:
            import cv2
            
            # Capture and analyze game state
            game_state = self.game_vision.get_game_state()
            
            print(f"✅ Screen capture successful")
            print(f"   Frame shape: {game_state['frame'].shape}")
            print(f"   Game state: {game_state.get('game_state', 'unknown')}")
            print(f"   Character detected: {'Yes' if game_state.get('character') else 'No'}")
            print(f"   Platforms detected: {len(game_state.get('platforms', []))}")
            print(f"   Power-ups detected: {len(game_state.get('power_ups', []))}")
            print(f"   Enemies detected: {len(game_state.get('enemies', []))}")
            print(f"   Estimated height: {game_state.get('height', 0)}")
            
            # Show debug visualization
            debug_frame = self.game_vision.draw_debug_overlay(
                game_state['frame'], game_state
            )
            
            cv2.imshow("Doodle Jump AI - Vision Test", debug_frame)
            print("👀 Debug window opened. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Vision test failed: {e}")
            
    def test_controls(self):
        """Test the game controller system."""
        print("🎮 Testing game controls...")
        print("⚠️  Make sure Doodle Jump is the active window!")
        
        for i in range(3, 0, -1):
            print(f"Starting control test in {i}...")
            time.sleep(1)
            
        try:
            # Test basic movements
            print("← Testing LEFT movement...")
            self.game_controller.move_left(0.5)
            time.sleep(0.5)
            
            print("→ Testing RIGHT movement...")
            self.game_controller.move_right(0.5)
            time.sleep(0.5)
            
            print("⏸️  Testing NO MOVEMENT...")
            self.game_controller.no_movement()
            time.sleep(0.5)
            
            print("🔄 Testing RESTART...")
            print("   (This will restart the game!)")
            time.sleep(2)
            # Uncomment to test restart: self.game_controller.restart_game()
            
            print("✅ Control test completed!")
            
        except Exception as e:
            print(f"❌ Control test failed: {e}")
        finally:
            self.game_controller.emergency_stop()
    
    def benchmark_performance(self):
        """Benchmark the AI's current performance."""
        print("🏃‍♂️ Running performance benchmark...")
        
        if not self.check_game_status():
            print("❌ Game not detected. Please start Doodle Jump first.")
            return
            
        # Load best model
        if os.path.exists("models/best_model.zip"):
            self.agent.load_model("models/best_model")
        else:
            print("⚠️  No trained model found. Benchmarking random agent.")
            
        # Run benchmark
        print("Running 5 benchmark games...")
        heights = []
        rewards = []
        
        for i in range(5):
            print(f"  Game {i+1}/5...")
            obs = self.agent.env.reset()[0]
            episode_reward = 0
            done = False
            max_height = 0
            
            while not done:
                action, _ = self.agent.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.agent.env.step(action)
                episode_reward += reward
                
                height = info.get('height', 0)
                if height > max_height:
                    max_height = height
                    
            heights.append(max_height)
            rewards.append(episode_reward)
            print(f"    Height: {max_height}, Reward: {episode_reward:.1f}")
            
        # Calculate stats
        avg_height = sum(heights) / len(heights)
        best_height = max(heights)
        avg_reward = sum(rewards) / len(rewards)
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Average Height: {avg_height:.1f}")
        print(f"   Best Height: {best_height}")
        print(f"   Average Reward: {avg_reward:.1f}")
        print(f"   Consistency: {(1 - (max(heights) - min(heights)) / max(heights)) * 100:.1f}%")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Doodle Jump AI Agent")
    parser.add_argument('command', choices=[
        'train', 'play', 'evaluate', 'progress', 'export', 'test-vision', 
        'test-controls', 'benchmark'
    ], help='Command to execute')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of episodes for training/playing/evaluation')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug visualization during play')
    parser.add_argument('--filename', type=str,
                       help='Filename for export command')
    
    args = parser.parse_args()
    
    # Initialize the AI
    try:
        ai = DoodleJumpAI()
    except Exception as e:
        print(f"❌ Failed to initialize AI: {e}")
        return 1
        
    # Execute command
    try:
        if args.command == 'train':
            ai.train_agent(args.episodes)
        elif args.command == 'play':
            ai.play_game(args.episodes, not args.no_debug)
        elif args.command == 'evaluate':
            ai.evaluate_agent(args.episodes)
        elif args.command == 'progress':
            ai.show_progress()
        elif args.command == 'export':
            ai.export_progress(args.filename)
        elif args.command == 'test-vision':
            ai.test_vision()
        elif args.command == 'test-controls':
            ai.test_controls()
        elif args.command == 'benchmark':
            ai.benchmark_performance()
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Command execution failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 