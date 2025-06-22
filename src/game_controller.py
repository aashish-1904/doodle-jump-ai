import pyautogui
import keyboard
import mouse
import time
import random
from typing import Dict, List, Tuple
import yaml

class DoodleController:
    """Controller for executing game actions in Doodle Jump."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.actions_config = self.config['actions']
        self.game_region = self.config['game']['game_region']
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01
        
        # Action mappings for Doodle Jump
        self.key_mappings = {
            'left': 'left',        # Left arrow key
            'right': 'right',      # Right arrow key  
            'tilt_left': 'a',      # Alternative left control
            'tilt_right': 'd',     # Alternative right control
            'shoot': 'space',      # For power-ups that can shoot
            'restart': 'r',        # Restart game
            'pause': 'p'           # Pause game
        }
        
        # Mouse controls (for mobile emulator or touch controls)
        self.mouse_control = {
            'left_side': self.game_region[2] // 4,      # Left side of screen
            'right_side': 3 * self.game_region[2] // 4,  # Right side of screen
            'center_y': self.game_region[3] // 2
        }
        
        self.current_action = None
        self.last_action_time = 0
        
    def move_left(self, duration: float = 0.1, method: str = "keyboard"):
        """Move character left."""
        if method == "keyboard":
            keyboard.press(self.key_mappings['left'])
            time.sleep(duration)
            keyboard.release(self.key_mappings['left'])
        elif method == "mouse":
            # Click/tap left side of screen
            x = self.game_region[0] + self.mouse_control['left_side']
            y = self.game_region[1] + self.mouse_control['center_y']
            pyautogui.click(x, y)
        elif method == "tilt":
            keyboard.press(self.key_mappings['tilt_left'])
            time.sleep(duration)
            keyboard.release(self.key_mappings['tilt_left'])
    
    def move_right(self, duration: float = 0.1, method: str = "keyboard"):
        """Move character right."""
        if method == "keyboard":
            keyboard.press(self.key_mappings['right'])
            time.sleep(duration)
            keyboard.release(self.key_mappings['right'])
        elif method == "mouse":
            # Click/tap right side of screen
            x = self.game_region[0] + self.mouse_control['right_side']
            y = self.game_region[1] + self.mouse_control['center_y']
            pyautogui.click(x, y)
        elif method == "tilt":
            keyboard.press(self.key_mappings['tilt_right'])
            time.sleep(duration)
            keyboard.release(self.key_mappings['tilt_right'])
    
    def no_movement(self):
        """Take no action (let character continue current trajectory)."""
        # Release any currently pressed keys
        for key in ['left', 'right', 'tilt_left', 'tilt_right']:
            try:
                keyboard.release(self.key_mappings[key])
            except:
                pass
        time.sleep(0.05)
    
    def shoot(self):
        """Shoot (for power-ups like propeller hat that can shoot)."""
        keyboard.press_and_release(self.key_mappings['shoot'])
    
    def restart_game(self):
        """Restart the game."""
        keyboard.press_and_release(self.key_mappings['restart'])
        time.sleep(1)  # Wait for game to restart
    
    def pause_game(self):
        """Pause/unpause the game."""
        keyboard.press_and_release(self.key_mappings['pause'])
    
    def move_to_platform(self, current_pos: Tuple[int, int], target_platform: Dict, urgency: float = 0.5):
        """Move towards a target platform intelligently."""
        if not target_platform:
            return
            
        char_x = current_pos[0]
        platform_center_x = target_platform['position'][0]
        platform_left = target_platform['left_x']
        platform_right = target_platform['right_x']
        
        # Calculate horizontal distance and direction
        distance_to_center = platform_center_x - char_x
        distance_to_platform = min(abs(platform_left - char_x), abs(platform_right - char_x))
        
        # Determine movement duration based on urgency and distance
        base_duration = 0.1
        duration = base_duration * urgency * min(2.0, distance_to_platform / 50)
        
        # Move towards platform
        if distance_to_center > 10:  # Need to move right
            self.move_right(duration)
        elif distance_to_center < -10:  # Need to move left
            self.move_left(duration)
        else:
            self.no_movement()  # Already well-positioned
    
    def avoid_enemy(self, current_pos: Tuple[int, int], enemy: Dict):
        """Move away from an enemy."""
        char_x = current_pos[0]
        enemy_x = enemy['position'][0]
        
        # Move away from enemy
        if enemy_x > char_x:
            self.move_left(0.2)  # Move left away from enemy
        else:
            self.move_right(0.2)  # Move right away from enemy
    
    def collect_power_up(self, current_pos: Tuple[int, int], power_up: Dict):
        """Move to collect a power-up."""
        char_x = current_pos[0]
        power_up_x = power_up['position'][0]
        
        # Move towards power-up
        if power_up_x > char_x:
            self.move_right(0.15)
        else:
            self.move_left(0.15)
    
    def execute_action(self, action: Dict):
        """Execute a complex action based on action dictionary."""
        action_type = action.get('type')
        
        if action_type == 'move_left':
            duration = action.get('duration', 0.1)
            method = action.get('method', 'keyboard')
            self.move_left(duration, method)
            
        elif action_type == 'move_right':
            duration = action.get('duration', 0.1)
            method = action.get('method', 'keyboard')
            self.move_right(duration, method)
            
        elif action_type == 'no_move':
            self.no_movement()
            
        elif action_type == 'shoot':
            self.shoot()
            
        elif action_type == 'restart':
            self.restart_game()
            
        elif action_type == 'pause':
            self.pause_game()
            
        elif action_type == 'move_to_platform':
            current_pos = action.get('current_position')
            target_platform = action.get('target_platform')
            urgency = action.get('urgency', 0.5)
            if current_pos and target_platform:
                self.move_to_platform(current_pos, target_platform, urgency)
                
        elif action_type == 'avoid_enemy':
            current_pos = action.get('current_position')
            enemy = action.get('enemy')
            if current_pos and enemy:
                self.avoid_enemy(current_pos, enemy)
                
        elif action_type == 'collect_power_up':
            current_pos = action.get('current_position')
            power_up = action.get('power_up')
            if current_pos and power_up:
                self.collect_power_up(current_pos, power_up)
    
    def get_optimal_action(self, game_state: Dict) -> Dict:
        """Determine optimal action based on current game state."""
        character = game_state.get('character')
        if not character:
            return {'type': 'no_move'}
            
        current_pos = character['position']
        platforms = game_state.get('platforms', [])
        enemies = game_state.get('enemies', [])
        power_ups = game_state.get('power_ups', [])
        
        # Priority 1: Avoid immediate danger
        for enemy in enemies:
            enemy_distance = abs(enemy['position'][0] - current_pos[0]) + abs(enemy['position'][1] - current_pos[1])
            if enemy_distance < 80:  # Very close enemy
                return {
                    'type': 'avoid_enemy',
                    'current_position': current_pos,
                    'enemy': enemy
                }
        
        # Priority 2: Collect nearby power-ups
        for power_up in power_ups:
            power_up_distance = abs(power_up['position'][0] - current_pos[0]) + abs(power_up['position'][1] - current_pos[1])
            if power_up_distance < 60:  # Close power-up
                return {
                    'type': 'collect_power_up',
                    'current_position': current_pos,
                    'power_up': power_up
                }
        
        # Priority 3: Move towards best platform
        char_y = current_pos[1]
        viable_platforms = [p for p in platforms if p['top_y'] < char_y and p['top_y'] > char_y - 200]
        
        if viable_platforms:
            # Choose best platform (closest horizontally, considering platform type)
            best_platform = None
            best_score = float('inf')
            
            for platform in viable_platforms:
                horizontal_distance = abs(platform['position'][0] - current_pos[0])
                vertical_distance = abs(platform['top_y'] - current_pos[1])
                
                # Score based on distance and platform type
                score = horizontal_distance + vertical_distance * 0.5
                
                # Bonus for special platforms
                if platform['type'] == 'blue':  # Moving platform
                    score *= 0.8  # Prefer moving platforms
                elif platform['type'] == 'brown':  # Breaking platform
                    score *= 1.2  # Avoid breaking platforms
                
                if score < best_score:
                    best_score = score
                    best_platform = platform
            
            if best_platform:
                # Calculate urgency based on character's falling speed and distance to platform
                urgency = min(1.0, 100.0 / (best_platform['top_y'] - char_y + 1))
                
                return {
                    'type': 'move_to_platform',
                    'current_position': current_pos,
                    'target_platform': best_platform,
                    'urgency': urgency
                }
        
        # Default: no movement
        return {'type': 'no_move'}
    
    def emergency_stop(self):
        """Release all keys and stop all actions."""
        for key in self.key_mappings.values():
            try:
                keyboard.release(key)
            except:
                pass
        self.current_action = None
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space for RL."""
        # Basic actions: left, right, no_move (3)
        return len(self.actions_config['movement'])
    
    def action_id_to_action(self, action_id: int) -> Dict:
        """Convert action ID to action dictionary for RL."""
        actions = self.actions_config['movement']
        
        if action_id < len(actions):
            action_name = actions[action_id]
            
            if action_name == 'left':
                return {'type': 'move_left', 'duration': 0.1}
            elif action_name == 'right':
                return {'type': 'move_right', 'duration': 0.1}
            elif action_name == 'no_move':
                return {'type': 'no_move'}
        
        return {'type': 'no_move'}
    
    def is_action_safe(self, action: Dict, game_state: Dict) -> bool:
        """Check if an action is safe to execute."""
        # Check if action would lead character into danger
        character = game_state.get('character')
        if not character:
            return True
            
        enemies = game_state.get('enemies', [])
        
        # Simple safety check: don't move towards very close enemies
        for enemy in enemies:
            enemy_distance = abs(enemy['position'][0] - character['position'][0])
            if enemy_distance < 50:  # Very close enemy
                action_type = action.get('type')
                if action_type == 'move_left' and enemy['position'][0] < character['position'][0]:
                    return False
                elif action_type == 'move_right' and enemy['position'][0] > character['position'][0]:
                    return False
        
        return True 