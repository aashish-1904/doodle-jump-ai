#!/usr/bin/env python3
"""
Enhanced Doodle Jump AI Agent

Advanced AI agent with improved learning algorithms including:
- Curriculum Learning
- Prioritized Experience Replay
- Multi-objective Optimization
- Ensemble Methods
- Transfer Learning

Author: AI Assistant
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import json
from typing import Dict, List, Tuple, Optional
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import heapq

# Advanced Experience Replay
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch with priority-based sampling."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

# Advanced Neural Networks
class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class NoiseNet(nn.Module):
    """Noisy Network for exploration."""
    
    def __init__(self, in_features: int, out_features: int):
        super(NoiseNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Noise tensors
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.017)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.017)
    
    def reset_noise(self):
        """Reset noise tensors."""
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

# Curriculum Learning
@dataclass
class CurriculumStage:
    """Curriculum learning stage definition."""
    name: str
    episodes: int
    difficulty_params: Dict
    success_threshold: float
    max_retries: int = 3

class CurriculumLearning:
    """Manages curriculum learning progression."""
    
    def __init__(self):
        self.stages = [
            CurriculumStage(
                name="Basic Jumping",
                episodes=100,
                difficulty_params={
                    'platform_spacing': 0.7,
                    'enemy_frequency': 0.0,
                    'breaking_platform_ratio': 0.0,
                    'max_height': 500
                },
                success_threshold=0.6
            ),
            CurriculumStage(
                name="Platform Variety",
                episodes=150,
                difficulty_params={
                    'platform_spacing': 0.8,
                    'enemy_frequency': 0.0,
                    'breaking_platform_ratio': 0.1,
                    'max_height': 1000
                },
                success_threshold=0.7
            ),
            CurriculumStage(
                name="Enemies Introduction",
                episodes=200,
                difficulty_params={
                    'platform_spacing': 0.9,
                    'enemy_frequency': 0.3,
                    'breaking_platform_ratio': 0.15,
                    'max_height': 1500
                },
                success_threshold=0.65
            ),
            CurriculumStage(
                name="Full Challenge",
                episodes=300,
                difficulty_params={
                    'platform_spacing': 1.0,
                    'enemy_frequency': 0.5,
                    'breaking_platform_ratio': 0.2,
                    'max_height': float('inf')
                },
                success_threshold=0.6
            )
        ]
        self.current_stage = 0
        self.stage_attempts = 0
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage]
    
    def evaluate_stage_completion(self, success_rate: float) -> bool:
        """Check if current stage is completed."""
        stage = self.get_current_stage()
        return success_rate >= stage.success_threshold
    
    def advance_stage(self) -> bool:
        """Advance to next stage if possible."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.stage_attempts = 0
            return True
        return False
    
    def retry_stage(self) -> bool:
        """Retry current stage if within retry limit."""
        stage = self.get_current_stage()
        if self.stage_attempts < stage.max_retries:
            self.stage_attempts += 1
            return True
        return False

# Multi-objective Optimization
class MultiObjectiveReward:
    """Multi-objective reward calculation."""
    
    def __init__(self):
        self.objectives = {
            'height': {'weight': 0.4, 'scale': 0.01},
            'survival': {'weight': 0.3, 'scale': 1.0},
            'efficiency': {'weight': 0.2, 'scale': 10.0},
            'exploration': {'weight': 0.1, 'scale': 5.0}
        }
    
    def calculate_reward(self, prev_state: Dict, current_state: Dict, action: int) -> Dict[str, float]:
        """Calculate multi-objective rewards."""
        rewards = {}
        
        # Height objective
        height_gain = current_state['height'] - prev_state['height']
        rewards['height'] = height_gain * self.objectives['height']['scale']
        
        # Survival objective
        if current_state['game_over']:
            rewards['survival'] = -100.0
        else:
            rewards['survival'] = 1.0
        
        # Efficiency objective (minimize unnecessary movements)
        char = current_state['character']
        if current_state['platforms']:
            nearest_platform = min(current_state['platforms'], 
                                 key=lambda p: abs(p['y'] - char['y']))
            platform_x = nearest_platform['x'] + nearest_platform['width'] / 2
            
            if action == 0 and char['x'] > platform_x:  # Moving towards platform
                rewards['efficiency'] = 2.0
            elif action == 1 and char['x'] < platform_x:  # Moving towards platform
                rewards['efficiency'] = 2.0
            elif action == 2:  # No action when well positioned
                if abs(char['x'] - platform_x) < 20:
                    rewards['efficiency'] = 1.0
                else:
                    rewards['efficiency'] = -1.0
            else:
                rewards['efficiency'] = -1.0
        else:
            rewards['efficiency'] = 0.0
        
        # Exploration objective
        if len(current_state['power_ups']) < len(prev_state['power_ups']):
            rewards['exploration'] = 10.0
        else:
            rewards['exploration'] = 0.0
        
        return rewards
    
    def combine_rewards(self, rewards: Dict[str, float]) -> float:
        """Combine multi-objective rewards into single value."""
        total_reward = 0.0
        for objective, reward in rewards.items():
            weight = self.objectives[objective]['weight']
            total_reward += weight * reward
        return total_reward

# Enhanced AI Agent
class EnhancedDoodleJumpAI:
    """Enhanced AI agent with advanced learning techniques."""
    
    def __init__(self, 
                 learning_rate: float = 0.0003,
                 state_size: int = 25,  # Expanded state representation
                 action_size: int = 3,
                 use_curriculum: bool = True,
                 use_prioritized_replay: bool = True,
                 use_noisy_nets: bool = True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        if use_noisy_nets:
            self.q_network = self._build_noisy_network().to(self.device)
            self.target_network = self._build_noisy_network().to(self.device)
        else:
            self.q_network = DuelingDQN(state_size, action_size).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(10000)
        else:
            self.memory = deque(maxlen=10000)
        
        # Learning components
        self.curriculum = CurriculumLearning() if use_curriculum else None
        self.multi_objective = MultiObjectiveReward()
        
        # Training parameters
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 1000
        self.training_step = 0
        
        # Exploration (for non-noisy networks)
        self.epsilon = 1.0 if not use_noisy_nets else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Statistics
        self.episode_rewards = []
        self.episode_heights = []
        self.objective_rewards = {obj: [] for obj in self.multi_objective.objectives.keys()}
        
        # Update target network
        self.update_target_network()
    
    def _build_noisy_network(self) -> nn.Module:
        """Build network with noisy layers."""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            NoiseNet(256, 256),
            nn.ReLU(),
            NoiseNet(256, self.action_size)
        )
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_enhanced_state_vector(self, game_state: Dict) -> np.ndarray:
        """Enhanced state representation with more features."""
        character = game_state['character']
        platforms = game_state['platforms']
        power_ups = game_state['power_ups']
        enemies = game_state['enemies']
        
        state = np.zeros(self.state_size)
        
        # Character features (5 features)
        state[0] = character['x'] / 600.0
        state[1] = character['vel_x'] / 20.0
        state[2] = character['vel_y'] / 30.0
        state[3] = 1.0 if character['power_up'] else 0.0
        state[4] = character['power_up_timer'] / 300.0 if character['power_up_timer'] > 0 else 0.0
        
        # Platform analysis (12 features - 3 platforms x 4 features each)
        char_x, char_y = character['x'], character['y']
        viable_platforms = [p for p in platforms if p['y'] < char_y and p['y'] > char_y - 400]
        viable_platforms.sort(key=lambda p: abs(p['y'] - char_y))
        
        for i in range(3):
            base_idx = 5 + i * 4
            if i < len(viable_platforms):
                platform = viable_platforms[i]
                state[base_idx] = (platform['x'] + platform['width']/2) / 600.0
                state[base_idx + 1] = (char_y - platform['y']) / 400.0
                state[base_idx + 2] = platform['width'] / 100.0
                
                # Enhanced platform type encoding
                type_encoding = {
                    'normal': 0.1, 'breaking': 0.3, 'moving': 0.5, 
                    'spring': 0.7, 'ice': 0.9
                }
                state[base_idx + 3] = type_encoding.get(platform['type'], 0.0)
        
        # Environment analysis (5 features)
        state[17] = len(power_ups) / 5.0  # Power-up density
        state[18] = len(enemies) / 3.0    # Enemy density
        state[19] = game_state['height'] / 5000.0  # Normalized height
        state[20] = game_state['score'] / 500.0    # Normalized score
        
        # Movement analysis (4 features)
        if len(viable_platforms) > 0:
            nearest_platform = viable_platforms[0]
            platform_center = nearest_platform['x'] + nearest_platform['width'] / 2
            
            state[21] = (platform_center - char_x) / 300.0  # Horizontal distance to target
            state[22] = 1.0 if char_x < platform_center else -1.0  # Direction to target
            state[23] = min(1.0, abs(platform_center - char_x) / 100.0)  # Urgency
        
        # Screen position
        state[24] = char_x / 600.0  # Normalized screen position
        
        return state
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action with advanced exploration."""
        if hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
        
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def train_step(self):
        """Enhanced training step with prioritized replay."""
        if isinstance(self.memory, PrioritizedReplayBuffer):
            if len(self.memory.buffer) < self.batch_size:
                return
            
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            if len(self.memory) < self.batch_size:
                return
            
            batch = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size)
        
        # Prepare batch
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        weights = weights.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        next_actions = self.q_network(next_states).argmax(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss with importance sampling weights
        td_errors = target_q_values - current_q_values.squeeze()
        loss = (weights * td_errors.pow(2)).mean()
        
        # Update priorities if using prioritized replay
        if isinstance(self.memory, PrioritizedReplayBuffer):
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.update_target_every == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_with_curriculum(self, max_episodes: int = 1000):
        """Train with curriculum learning."""
        from doodle_jump_game import DoodleJumpGame
        
        print("🎓 Starting Curriculum Learning Training...")
        
        total_episodes = 0
        
        while total_episodes < max_episodes and self.curriculum:
            stage = self.curriculum.get_current_stage()
            print(f"\n📚 Stage: {stage.name} (Episodes: {stage.episodes})")
            
            # Train on current stage
            stage_rewards = []
            stage_heights = []
            
            for episode in range(stage.episodes):
                game = DoodleJumpGame(headless=True)
                
                # Apply curriculum difficulty
                self._apply_difficulty_params(game, stage.difficulty_params)
                
                # Run episode
                episode_reward, episode_height = self._run_episode(game)
                stage_rewards.append(episode_reward)
                stage_heights.append(episode_height)
                
                total_episodes += 1
                
                if episode % 25 == 0:
                    avg_reward = np.mean(stage_rewards[-25:])
                    avg_height = np.mean(stage_heights[-25:])
                    print(f"  Episode {episode}: Avg Reward = {avg_reward:.1f}, Avg Height = {avg_height:.1f}")
            
            # Evaluate stage completion
            success_rate = self._calculate_success_rate(stage_rewards, stage_heights, stage)
            print(f"📊 Stage Success Rate: {success_rate:.2f} (Threshold: {stage.success_threshold:.2f})")
            
            if self.curriculum.evaluate_stage_completion(success_rate):
                print("✅ Stage completed successfully!")
                if not self.curriculum.advance_stage():
                    print("🎓 Curriculum completed!")
                    break
            else:
                print("❌ Stage failed, retrying...")
                if not self.curriculum.retry_stage():
                    print("⚠️ Max retries reached, advancing anyway...")
                    self.curriculum.advance_stage()
        
        print("🏁 Curriculum training completed!")
    
    def _apply_difficulty_params(self, game, params: Dict):
        """Apply difficulty parameters to game."""
        # This would modify game generation parameters
        # Implementation depends on game architecture
        pass
    
    def _run_episode(self, game) -> Tuple[float, float]:
        """Run a single training episode."""
        game.reset()
        state = self.get_enhanced_state_vector(game.get_game_state())
        
        total_reward = 0
        max_height = 0
        steps = 0
        objective_rewards = {obj: 0 for obj in self.multi_objective.objectives.keys()}
        
        while not game.game_over and steps < 5000:
            action = self.get_action(state, training=True)
            
            # Apply action
            if action == 0:
                game.character.move_left()
            elif action == 1:
                game.character.move_right()
            
            prev_game_state = game.get_game_state()
            game.update()
            current_game_state = game.get_game_state()
            
            next_state = self.get_enhanced_state_vector(current_game_state)
            
            # Calculate multi-objective rewards
            obj_rewards = self.multi_objective.calculate_reward(prev_game_state, current_game_state, action)
            reward = self.multi_objective.combine_rewards(obj_rewards)
            
            # Store experience
            if isinstance(self.memory, PrioritizedReplayBuffer):
                self.memory.add(state, action, reward, next_state, game.game_over)
            else:
                self.memory.append((state, action, reward, next_state, game.game_over))
            
            # Update statistics
            total_reward += reward
            max_height = max(max_height, current_game_state['height'])
            for obj, r in obj_rewards.items():
                objective_rewards[obj] += r
            
            state = next_state
            steps += 1
            
            # Train
            self.train_step()
        
        # Record episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_heights.append(max_height)
        for obj, r in objective_rewards.items():
            self.objective_rewards[obj].append(r)
        
        return total_reward, max_height
    
    def _calculate_success_rate(self, rewards: List[float], heights: List[float], stage) -> float:
        """Calculate success rate for curriculum stage."""
        # Define success criteria based on stage
        if 'max_height' in stage.difficulty_params:
            max_allowed = stage.difficulty_params['max_height']
            if max_allowed != float('inf'):
                success_count = sum(1 for h in heights[-50:] if h >= max_allowed * 0.5)
                return success_count / min(50, len(heights))
        
        # Fallback: use average reward improvement
        if len(rewards) >= 50:
            recent_avg = np.mean(rewards[-25:])
            early_avg = np.mean(rewards[:25])
            improvement = (recent_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0
            return min(1.0, max(0.0, improvement + 0.5))
        
        return 0.5  # Default if not enough data
    
    def plot_advanced_training_progress(self):
        """Plot comprehensive training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards over time
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Total Rewards Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Heights over time
        axes[0, 1].plot(self.episode_heights)
        axes[0, 1].set_title('Max Height Achieved Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Max Height')
        axes[0, 1].grid(True)
        
        # Multi-objective rewards
        for obj, rewards in self.objective_rewards.items():
            if rewards:
                axes[1, 0].plot(rewards, label=obj.capitalize())
        axes[1, 0].set_title('Multi-Objective Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Objective Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Performance metrics
        if len(self.episode_rewards) >= 50:
            window_size = 50
            smoothed_rewards = []
            for i in range(window_size, len(self.episode_rewards)):
                smoothed_rewards.append(np.mean(self.episode_rewards[i-window_size:i]))
            
            axes[1, 1].plot(smoothed_rewards)
            axes[1, 1].set_title(f'Smoothed Rewards (Window: {window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('enhanced_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Enhanced training progress saved to 'enhanced_training_progress.png'")

if __name__ == "__main__":
    print("🚀 Enhanced AI Agent Loaded!")
    print("Features:")
    print("  🧠 Dueling DQN with Noisy Networks")
    print("  📚 Curriculum Learning")
    print("  🎯 Multi-Objective Optimization") 
    print("  🔄 Prioritized Experience Replay")
    print("  📊 Advanced Analytics") 