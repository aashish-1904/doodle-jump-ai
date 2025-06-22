import json
import time
import os
from typing import Dict, List, Any
from datetime import datetime
import yaml

class DoodleProgressTracker:
    """Tracks and manages Doodle Jump AI progress and achievements."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.progress_config = self.config['progress']
        self.save_path = self.progress_config['save_path']
        
        # Initialize progress data
        self.progress_data = {
            'game_stats': {
                'high_score': 0,
                'best_height': 0,
                'average_height': 0,
                'games_played': 0,
                'total_playtime': 0,
                'platforms_landed_total': 0,
                'power_ups_collected_total': 0,
                'enemies_avoided_total': 0
            },
            'achievements': {},
            'game_history': [],
            'training_metrics': {
                'episodes_completed': 0,
                'average_performance': 0,
                'learning_curve': [],
                'best_episode_reward': 0,
                'training_hours': 0
            },
            'death_statistics': {
                'game_over': 0,
                'fell_off_screen': 0,
                'time_limit': 0,
                'enemy_collision': 0
            },
            'last_updated': None
        }
        
        # Load existing progress
        self.load_progress()
        
        # Achievement definitions
        self.achievements = {
            'first_jump': {'name': 'First Jump', 'description': 'Complete your first game', 'unlocked': False},
            'height_100': {'name': 'Sky Walker', 'description': 'Reach height of 100', 'unlocked': False},
            'height_500': {'name': 'Cloud Jumper', 'description': 'Reach height of 500', 'unlocked': False},
            'height_1000': {'name': 'Space Explorer', 'description': 'Reach height of 1000', 'unlocked': False},
            'platform_master': {'name': 'Platform Master', 'description': 'Land on 100 platforms in total', 'unlocked': False},
            'power_collector': {'name': 'Power Collector', 'description': 'Collect 50 power-ups', 'unlocked': False},
            'training_dedication': {'name': 'Training Dedication', 'description': 'Complete 1000 training episodes', 'unlocked': False},
            'consistency': {'name': 'Consistency Champion', 'description': 'Play 100 games', 'unlocked': False}
        }
        
    def load_progress(self):
        """Load progress from save file."""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    saved_data = json.load(f)
                    # Merge with default structure to handle new fields
                    self._merge_progress_data(saved_data)
                print(f"Progress loaded from {self.save_path}")
            else:
                print("No existing progress found. Starting fresh.")
        except Exception as e:
            print(f"Error loading progress: {e}. Starting fresh.")
            
    def _merge_progress_data(self, saved_data: Dict):
        """Merge saved data with current structure."""
        def deep_merge(default: Dict, saved: Dict):
            for key, value in saved.items():
                if key in default:
                    if isinstance(default[key], dict) and isinstance(value, dict):
                        deep_merge(default[key], value)
                    else:
                        default[key] = value
                        
        deep_merge(self.progress_data, saved_data)
        
    def save_progress(self):
        """Save current progress to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Update timestamp
            self.progress_data['last_updated'] = datetime.now().isoformat()
            
            # Save to file
            with open(self.save_path, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
                
            # Create backup periodically
            if self.progress_data['game_stats']['games_played'] % self.progress_config['backup_frequency'] == 0:
                backup_path = f"{self.save_path}.backup_{int(time.time())}"
                with open(backup_path, 'w') as f:
                    json.dump(self.progress_data, f, indent=2)
                    
        except Exception as e:
            print(f"Error saving progress: {e}")
            
    def update_game_stats(self, episode_stats: Dict):
        """Update player statistics after a game."""
        stats = self.progress_data['game_stats']
        
        # Update basic stats
        stats['games_played'] += 1
        
        # Update height records
        max_height = episode_stats.get('max_height', 0)
        if max_height > stats['best_height']:
            stats['best_height'] = max_height
            print(f"🎉 NEW HEIGHT RECORD: {max_height}!")
        
        # Update running averages
        total_games = stats['games_played']
        current_avg = stats['average_height']
        stats['average_height'] = ((current_avg * (total_games - 1)) + max_height) / total_games
        
        # Update totals
        stats['platforms_landed_total'] += episode_stats.get('platforms_landed', 0)
        stats['power_ups_collected_total'] += episode_stats.get('power_ups_collected', 0)
        stats['enemies_avoided_total'] += episode_stats.get('enemies_avoided', 0)
        stats['total_playtime'] += episode_stats.get('game_duration', 0)
        
        # Update death statistics
        death_cause = episode_stats.get('death_cause', 'unknown')
        if death_cause in self.progress_data['death_statistics']:
            self.progress_data['death_statistics'][death_cause] += 1
        
        # Record game in history
        game_record = {
            'timestamp': datetime.now().isoformat(),
            'stats': episode_stats.copy(),
            'was_record': max_height == stats['best_height']
        }
        self.progress_data['game_history'].append(game_record)
        
        # Keep only last 100 games in history
        if len(self.progress_data['game_history']) > 100:
            self.progress_data['game_history'] = self.progress_data['game_history'][-100:]
            
        # Check achievements
        self._check_achievements()
        
        # Auto-save
        self.save_progress()
        
    def update_training_stats(self, episode_reward: float, episode_length: int, training_time: float = 0):
        """Update training metrics."""
        metrics = self.progress_data['training_metrics']
        
        metrics['episodes_completed'] += 1
        metrics['training_hours'] += training_time / 3600  # Convert to hours
        
        # Update learning curve
        metrics['learning_curve'].append({
            'episode': metrics['episodes_completed'],
            'reward': episode_reward,
            'length': episode_length,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 episodes in learning curve
        if len(metrics['learning_curve']) > 1000:
            metrics['learning_curve'] = metrics['learning_curve'][-1000:]
            
        # Update best performance
        if episode_reward > metrics['best_episode_reward']:
            metrics['best_episode_reward'] = episode_reward
            print(f"🏆 NEW TRAINING RECORD: {episode_reward:.2f} reward!")
            
        # Calculate rolling average performance
        recent_episodes = metrics['learning_curve'][-50:]  # Last 50 episodes
        if recent_episodes:
            metrics['average_performance'] = sum(ep['reward'] for ep in recent_episodes) / len(recent_episodes)
    
    def _check_achievements(self):
        """Check and unlock achievements."""
        stats = self.progress_data['game_stats']
        training_stats = self.progress_data['training_metrics']
        
        # First jump achievement
        if stats['games_played'] >= 1 and not self.achievements['first_jump']['unlocked']:
            self._unlock_achievement('first_jump')
            
        # Height achievements
        if stats['best_height'] >= 100 and not self.achievements['height_100']['unlocked']:
            self._unlock_achievement('height_100')
        if stats['best_height'] >= 500 and not self.achievements['height_500']['unlocked']:
            self._unlock_achievement('height_500')
        if stats['best_height'] >= 1000 and not self.achievements['height_1000']['unlocked']:
            self._unlock_achievement('height_1000')
            
        # Platform master achievement
        if stats['platforms_landed_total'] >= 100 and not self.achievements['platform_master']['unlocked']:
            self._unlock_achievement('platform_master')
            
        # Power collector achievement
        if stats['power_ups_collected_total'] >= 50 and not self.achievements['power_collector']['unlocked']:
            self._unlock_achievement('power_collector')
            
        # Training dedication achievement
        if training_stats['episodes_completed'] >= 1000 and not self.achievements['training_dedication']['unlocked']:
            self._unlock_achievement('training_dedication')
            
        # Consistency achievement
        if stats['games_played'] >= 100 and not self.achievements['consistency']['unlocked']:
            self._unlock_achievement('consistency')
                
    def _unlock_achievement(self, achievement_id: str):
        """Unlock an achievement."""
        if achievement_id in self.achievements:
            self.achievements[achievement_id]['unlocked'] = True
            self.progress_data['achievements'][achievement_id] = {
                'unlocked_at': datetime.now().isoformat(),
                'name': self.achievements[achievement_id]['name'],
                'description': self.achievements[achievement_id]['description']
            }
            print(f"🏆 Achievement Unlocked: {self.achievements[achievement_id]['name']}")
            
    def get_progress_summary(self) -> Dict:
        """Get a summary of current progress."""
        stats = self.progress_data['game_stats']
        training_stats = self.progress_data['training_metrics']
        
        # Calculate time played in hours
        hours_played = stats['total_playtime'] / 3600
        
        # Count achievements
        unlocked_achievements = len([a for a in self.achievements.values() if a['unlocked']])
        
        # Calculate improvement trend
        recent_games = self.progress_data['game_history'][-10:]
        if len(recent_games) >= 2:
            recent_avg = sum(game['stats'].get('max_height', 0) for game in recent_games) / len(recent_games)
            improvement_trend = "↗️" if recent_avg > stats['average_height'] else "↘️"
        else:
            improvement_trend = "➡️"
        
        return {
            'high_score': stats['best_height'],
            'average_height': round(stats['average_height'], 1),
            'games_played': stats['games_played'],
            'hours_played': round(hours_played, 1),
            'training_hours': round(training_stats['training_hours'], 1),
            'platforms_landed': stats['platforms_landed_total'],
            'power_ups_collected': stats['power_ups_collected_total'],
            'achievements_unlocked': f"{unlocked_achievements}/{len(self.achievements)}",
            'training_episodes': training_stats['episodes_completed'],
            'average_performance': round(training_stats['average_performance'], 2),
            'improvement_trend': improvement_trend
        }
        
    def print_progress_report(self):
        """Print a detailed progress report."""
        summary = self.get_progress_summary()
        
        print("\n" + "="*50)
        print("🦘 DOODLE JUMP AI PROGRESS REPORT")
        print("="*50)
        print(f"🏆 High Score: {summary['high_score']}")
        print(f"📊 Average Height: {summary['average_height']}")
        print(f"🎮 Games Played: {summary['games_played']}")
        print(f"⏱️  Time Played: {summary['hours_played']} hours")
        print(f"🤖 Training Time: {summary['training_hours']} hours")
        print(f"🪜 Platforms Landed: {summary['platforms_landed']}")
        print(f"⚡ Power-ups Collected: {summary['power_ups_collected']}")
        print(f"🏅 Achievements: {summary['achievements_unlocked']}")
        print(f"📈 Training Episodes: {summary['training_episodes']}")
        print(f"📊 AI Performance: {summary['average_performance']}")
        print(f"📈 Trend: {summary['improvement_trend']}")
        
        # Show death statistics
        death_stats = self.progress_data['death_statistics']
        print(f"\n💀 Death Causes:")
        total_deaths = sum(death_stats.values())
        if total_deaths > 0:
            for cause, count in death_stats.items():
                percentage = (count / total_deaths) * 100
                print(f"   {cause.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Show recent achievements
        recent_achievements = []
        for achievement_id, achievement_data in self.progress_data['achievements'].items():
            if achievement_id in self.achievements:
                recent_achievements.append(achievement_data['name'])
                
        if recent_achievements:
            print(f"\n🏆 Recent Achievements:")
            for achievement in recent_achievements[-3:]:  # Show last 3 achievements
                print(f"   • {achievement}")
                
        print("="*50)
        
    def get_learning_curve_data(self) -> List[Dict]:
        """Get learning curve data for visualization."""
        return self.progress_data['training_metrics']['learning_curve']
        
    def export_stats(self, filename: str = None):
        """Export statistics to a file."""
        if filename is None:
            filename = f"doodle_stats_{int(time.time())}.json"
            
        export_data = {
            'summary': self.get_progress_summary(),
            'full_data': self.progress_data,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"📄 Statistics exported to {filename}") 