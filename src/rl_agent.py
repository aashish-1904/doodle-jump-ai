import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import cv2
from typing import Dict, List, Tuple, Any
import yaml
import time

class DoodleJumpEnv(gym.Env):
    """Custom Gym environment for Doodle Jump game."""
    
    def __init__(self, game_vision, game_controller, config_path="config.yaml"):
        super(DoodleJumpEnv, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.game_vision = game_vision
        self.game_controller = game_controller
        self.reward_config = self.config['rewards']
        
        # Define action space (discrete actions: left, right, no_move)
        self.action_space = spaces.Discrete(game_controller.get_action_space_size())
        
        # Define observation space (processed game screen)
        screen_height, screen_width = 84, 84
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(screen_height, screen_width, 3), 
            dtype=np.uint8
        )
        
        # Game state tracking
        self.last_game_state = None
        self.episode_start_time = None
        self.max_height_reached = 0
        self.episode_stats = self.reset_episode_stats()
        
    def reset_episode_stats(self):
        """Reset episode statistics."""
        return {
            'max_height': 0,
            'platforms_landed': 0,
            'power_ups_collected': 0,
            'enemies_avoided': 0,
            'game_duration': 0,
            'death_cause': 'unknown'
        }
        
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.episode_stats = self.reset_episode_stats()
        self.episode_start_time = time.time()
        self.max_height_reached = 0
        
        # Wait for game to be ready
        self._wait_for_game_ready()
        
        # Get initial game state
        self.last_game_state = self.game_vision.get_game_state()
        observation = self._process_observation(self.last_game_state)
        
        return observation, {}
    
    def step(self, action):
        """Execute action and return new state, reward, done, info."""
        # Convert action ID to action dictionary
        action_dict = self.game_controller.action_id_to_action(action)
        
        # Execute action
        self.game_controller.execute_action(action_dict)
        
        # Small delay to allow game to respond
        time.sleep(0.05)
        
        # Get new game state
        current_game_state = self.game_vision.get_game_state()
        
        # Calculate reward
        reward = self._calculate_reward(current_game_state, action_dict)
        
        # Check if episode is done
        done = self._check_episode_done(current_game_state)
        
        # Process observation
        observation = self._process_observation(current_game_state)
        
        # Update tracking
        self._update_episode_stats(current_game_state)
        self.last_game_state = current_game_state
        
        info = {
            'episode_stats': self.episode_stats.copy(),
            'game_state': current_game_state,
            'height': current_game_state.get('height', 0)
        }
        
        return observation, reward, done, False, info
    
    def _process_observation(self, game_state: Dict) -> np.ndarray:
        """Process game state into observation for neural network."""
        frame = game_state['frame']
        
        # Resize frame to standard size
        processed_frame = cv2.resize(frame, (84, 84))
        
        # Convert to RGB
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        return processed_frame
    
    def _calculate_reward(self, current_state: Dict, action: Dict) -> float:
        """Calculate reward based on game state changes."""
        reward = 0.0
        
        if self.last_game_state is None:
            return reward
            
        # Height-based rewards
        current_height = current_state.get('height', 0)
        last_height = self.last_game_state.get('height', 0)
        
        height_gain = current_height - last_height
        if height_gain > 0:
            reward += self.reward_config['height_gain'] * height_gain
            
        # Update max height and give bonus
        if current_height > self.max_height_reached:
            height_increase = current_height - self.max_height_reached
            reward += height_increase * 2  # Bonus for new height records
            self.max_height_reached = current_height
            self.episode_stats['max_height'] = current_height
        
        # Platform landing detection
        current_character = current_state.get('character')
        last_character = self.last_game_state.get('character')
        
        if current_character and last_character:
            # Check if character landed on a platform (y position decreased significantly)
            y_change = current_character['position'][1] - last_character['position'][1]
            if y_change < -20:  # Character moved up significantly (landed on platform)
                reward += self.reward_config['platform_land']
                self.episode_stats['platforms_landed'] += 1
        
        # Power-up collection detection
        current_power_ups = len(current_state.get('power_ups', []))
        last_power_ups = len(self.last_game_state.get('power_ups', []))
        
        if current_power_ups < last_power_ups:  # Power-up disappeared (collected)
            reward += self.reward_config['power_up_collect']
            self.episode_stats['power_ups_collected'] += 1
        
        return reward
    
    def _check_episode_done(self, game_state: Dict) -> bool:
        """Check if episode should end."""
        # Check game state
        if game_state.get('game_state') == 'game_over':
            self.episode_stats['death_cause'] = 'game_over'
            return True
            
        # Check if character is missing (fell off screen)
        if not game_state.get('character'):
            self.episode_stats['death_cause'] = 'fell_off_screen'
            return True
            
        # Check for maximum episode length
        if time.time() - self.episode_start_time > 300:  # 5 minutes max
            self.episode_stats['death_cause'] = 'time_limit'
            return True
            
        return False
    
    def _update_episode_stats(self, game_state: Dict):
        """Update episode statistics."""
        self.episode_stats['game_duration'] = time.time() - self.episode_start_time
        
        current_height = game_state.get('height', 0)
        if current_height > self.episode_stats['max_height']:
            self.episode_stats['max_height'] = current_height
    
    def _wait_for_game_ready(self):
        """Wait for game to be in a playable state."""
        max_wait_time = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            game_state = self.game_vision.get_game_state()
            if game_state.get('game_state') == 'playing' and game_state.get('character'):
                return
            
            # Try to restart if in game over state
            if game_state.get('game_state') == 'game_over':
                self.game_controller.restart_game()
                time.sleep(2)
            
            time.sleep(0.5)

class DoodleJumpAgent:
    """Main RL agent for playing Doodle Jump."""
    
    def __init__(self, game_vision, game_controller, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.game_vision = game_vision
        self.game_controller = game_controller
        
        # Create environment
        self.env = DoodleJumpEnv(game_vision, game_controller, config_path)
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Create PPO model
        self.model = PPO(
            "CnnPolicy",
            self.vec_env,
            learning_rate=self.config['rl']['learning_rate'],
            batch_size=self.config['rl']['batch_size'],
            gamma=self.config['rl']['gamma'],
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
        
        # Training stats
        self.training_stats = {
            'episodes_trained': 0,
            'best_height': 0,
            'average_height': 0,
            'total_rewards': [],
            'episode_lengths': []
        }
        
    def train(self, total_episodes: int = None):
        """Train the agent."""
        if total_episodes is None:
            total_episodes = self.config['training']['episodes']
            
        print(f"Starting Doodle Jump AI training for {total_episodes} episodes...")
        
        save_freq = self.config['training']['save_frequency']
        eval_freq = self.config['training']['evaluation_frequency']
        
        for episode in range(total_episodes):
            # Train for one episode
            self.model.learn(total_timesteps=1000)
            
            self.training_stats['episodes_trained'] += 1
            
            # Save model periodically
            if episode % save_freq == 0 and episode > 0:
                self.save_model(f"models/ppo_doodle_episode_{episode}")
                
            # Evaluate performance periodically
            if episode % eval_freq == 0 and episode > 0:
                avg_height, avg_reward = self.evaluate(num_episodes=3)
                print(f"Episode {episode}: Avg Height = {avg_height:.1f}, Avg Reward = {avg_reward:.2f}")
                
                if avg_height > self.training_stats['best_height']:
                    self.training_stats['best_height'] = avg_height
                    self.save_model("models/best_model")
                    
                self.training_stats['average_height'] = avg_height
                    
        print("Training completed!")
        
    def evaluate(self, num_episodes: int = 5) -> Tuple[float, float]:
        """Evaluate the agent's performance."""
        total_heights = []
        total_rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()[0]
            episode_reward = 0
            done = False
            max_height = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                
                height = info.get('height', 0)
                if height > max_height:
                    max_height = height
                
            total_heights.append(max_height)
            total_rewards.append(episode_reward)
            
        return np.mean(total_heights), np.mean(total_rewards)
    
    def play(self, num_episodes: int = 1, render_debug: bool = True):
        """Play the game using the trained agent."""
        for episode in range(num_episodes):
            print(f"Playing Doodle Jump episode {episode + 1}/{num_episodes}")
            
            obs = self.env.reset()[0]
            episode_reward = 0
            done = False
            step_count = 0
            max_height = 0
            
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Execute action
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Track height
                height = info.get('height', 0)
                if height > max_height:
                    max_height = height
                
                # Optional: show debug visualization
                if render_debug and step_count % 10 == 0:
                    game_state = info['game_state']
                    debug_frame = self.game_vision.draw_debug_overlay(
                        game_state['frame'], game_state
                    )
                    cv2.imshow("Doodle Jump AI Debug", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            print(f"Episode {episode + 1} completed:")
            print(f"  Max Height: {max_height}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Stats: {info['episode_stats']}")
            
        if render_debug:
            cv2.destroyAllWindows()
    
    def save_model(self, path: str):
        """Save the trained model."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load a trained model."""
        self.model = PPO.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")
        
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats.copy() 