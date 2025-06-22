import cv2
import numpy as np
import mss
from PIL import Image
import time
from typing import Dict, List, Tuple, Optional
import yaml

class DoodleVision:
    """Computer vision system for Doodle Jump game state detection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.sct = mss.mss()
        self.monitor = self.config['capture']['monitor']
        self.game_region = self.config['game']['game_region']
        self.vision_config = self.config['vision']
        
        # Template storage for UI elements
        self.templates = {}
        self.load_templates()
        
        # Platform colors for detection
        self.platform_colors = self.vision_config['platform_colors']
        
    def load_templates(self):
        """Load template images for game element detection."""
        template_paths = {
            'doodle_character': 'assets/templates/doodle.png',
            'score_area': 'assets/templates/score.png',
            'game_over': 'assets/templates/game_over.png',
            'power_ups': {
                'spring': 'assets/templates/spring.png',
                'rocket': 'assets/templates/rocket.png',
                'propeller': 'assets/templates/propeller.png',
                'jetpack': 'assets/templates/jetpack.png'
            },
            'enemies': {
                'monster': 'assets/templates/monster.png',
                'hole': 'assets/templates/hole.png',
                'ufo': 'assets/templates/ufo.png'
            }
        }
        
        for name, path in template_paths.items():
            try:
                if isinstance(path, dict):
                    self.templates[name] = {}
                    for sub_name, sub_path in path.items():
                        try:
                            import os
                            if os.path.exists(sub_path):
                                self.templates[name][sub_name] = cv2.imread(sub_path, 0)
                        except Exception as e:
                            print(f"Warning: Could not load template {name}/{sub_name}: {e}")
                else:
                    import os
                    if os.path.exists(path):
                        self.templates[name] = cv2.imread(path, 0)
            except Exception as e:
                print(f"Warning: Could not load template {name}: {e}")
    
    def capture_screen(self) -> np.ndarray:
        """Capture the current game screen."""
        monitor_config = {
            "top": self.game_region[1],
            "left": self.game_region[0], 
            "width": self.game_region[2] - self.game_region[0],
            "height": self.game_region[3] - self.game_region[1]
        }
        
        screenshot = self.sct.grab(monitor_config)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def detect_platforms(self, frame: np.ndarray) -> List[Dict]:
        """Detect platforms in the game."""
        platforms = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for platform_type, color_range in self.platform_colors.items():
            lower = np.array(color_range[0])
            upper = np.array(color_range[1])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.vision_config['contour_min_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter platform-like shapes (wider than tall)
                    if w > h and w > 30:  # Minimum platform width
                        platforms.append({
                            'type': platform_type,
                            'position': (x + w//2, y + h//2),
                            'bbox': (x, y, w, h),
                            'area': area,
                            'top_y': y,
                            'bottom_y': y + h,
                            'left_x': x,
                            'right_x': x + w
                        })
        
        # Sort platforms by Y position (top to bottom)
        platforms.sort(key=lambda p: p['top_y'])
        return platforms
    
    def detect_character(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect the Doodle character position."""
        # Method 1: Template matching if template is available
        if 'doodle_character' in self.templates and self.templates['doodle_character'] is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, self.templates['doodle_character'], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > self.vision_config['template_matching_threshold']:
                x, y = max_loc
                h, w = self.templates['doodle_character'].shape
                return {
                    'position': (x + w//2, y + h//2),
                    'bbox': (x, y, w, h),
                    'confidence': max_val
                }
        
        # Method 2: Color-based detection (look for character colors)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Doodle character color ranges (adjust based on actual game)
        character_colors = [
            ([20, 100, 100], [30, 255, 255]),  # Yellow/green doodle
            ([35, 50, 50], [85, 255, 255])     # Green doodle
        ]
        
        for lower, upper in character_colors:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # Character size range
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's character-like (roughly square/vertical)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 2.0:
                        return {
                            'position': (x + w//2, y + h//2),
                            'bbox': (x, y, w, h),
                            'confidence': 0.8
                        }
        
        return None
    
    def detect_power_ups(self, frame: np.ndarray) -> List[Dict]:
        """Detect power-ups in the game."""
        power_ups = []
        
        if 'power_ups' not in self.templates:
            return power_ups
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for power_type, template in self.templates['power_ups'].items():
            if template is not None:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.vision_config['template_matching_threshold'])
                
                for pt in zip(*locations[::-1]):
                    h, w = template.shape
                    power_ups.append({
                        'type': power_type,
                        'position': (pt[0] + w//2, pt[1] + h//2),
                        'bbox': (pt[0], pt[1], w, h),
                        'confidence': result[pt[1], pt[0]]
                    })
        
        return power_ups
    
    def detect_enemies(self, frame: np.ndarray) -> List[Dict]:
        """Detect enemies and obstacles."""
        enemies = []
        
        if 'enemies' not in self.templates:
            return enemies
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for enemy_type, template in self.templates['enemies'].items():
            if template is not None:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.vision_config['template_matching_threshold'])
                
                for pt in zip(*locations[::-1]):
                    h, w = template.shape
                    enemies.append({
                        'type': enemy_type,
                        'position': (pt[0] + w//2, pt[1] + h//2),
                        'bbox': (pt[0], pt[1], w, h),
                        'confidence': result[pt[1], pt[0]]
                    })
        
        return enemies
    
    def detect_score(self, frame: np.ndarray) -> Optional[int]:
        """Extract current score from the game."""
        # This would need OCR or score area analysis
        # For now, return None (could be implemented with pytesseract)
        return None
    
    def detect_game_state(self, frame: np.ndarray) -> str:
        """Detect current game state (playing, game_over, menu)."""
        # Check for game over screen
        if 'game_over' in self.templates and self.templates['game_over'] is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, self.templates['game_over'], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > self.vision_config['template_matching_threshold']:
                return 'game_over'
        
        # Check if character is detected (game is playing)
        character = self.detect_character(frame)
        if character:
            return 'playing'
        
        return 'menu'
    
    def estimate_height(self, frame: np.ndarray, character_pos: Optional[Dict]) -> int:
        """Estimate current height in the game."""
        if not character_pos:
            return 0
            
        # Height is inversely related to Y position (higher Y = lower in game)
        # This is a rough estimation - could be improved with score detection
        height, width = frame.shape[:2]
        char_y = character_pos['position'][1]
        
        # Estimate height based on character position
        estimated_height = max(0, height - char_y)
        return estimated_height
    
    def get_game_state(self) -> Dict:
        """Get comprehensive game state information."""
        frame = self.capture_screen()
        
        # Detect all game elements
        character = self.detect_character(frame)
        platforms = self.detect_platforms(frame)
        power_ups = self.detect_power_ups(frame)
        enemies = self.detect_enemies(frame)
        game_state = self.detect_game_state(frame)
        score = self.detect_score(frame)
        height = self.estimate_height(frame, character)
        
        return {
            'frame': frame,
            'timestamp': time.time(),
            'character': character,
            'platforms': platforms,
            'power_ups': power_ups,
            'enemies': enemies,
            'game_state': game_state,
            'score': score,
            'height': height
        }
    
    def get_next_platforms(self, platforms: List[Dict], character_pos: Dict, num_platforms: int = 5) -> List[Dict]:
        """Get the next platforms above the character."""
        if not character_pos:
            return []
            
        char_y = character_pos['position'][1]
        
        # Filter platforms above the character
        above_platforms = [p for p in platforms if p['top_y'] < char_y]
        
        # Sort by distance to character (closest first)
        above_platforms.sort(key=lambda p: abs(p['top_y'] - char_y))
        
        return above_platforms[:num_platforms]
    
    def draw_debug_overlay(self, frame: np.ndarray, game_state: Dict) -> np.ndarray:
        """Draw debug information on the frame."""
        debug_frame = frame.copy()
        
        # Draw character
        if game_state.get('character'):
            char = game_state['character']
            x, y, w, h = char['bbox']
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(debug_frame, "DOODLE", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw platforms
        platform_colors = {
            'green': (0, 255, 0),
            'brown': (0, 165, 255),
            'blue': (255, 0, 0),
            'white': (255, 255, 255)
        }
        
        for platform in game_state.get('platforms', []):
            x, y, w, h = platform['bbox']
            color = platform_colors.get(platform['type'], (128, 128, 128))
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_frame, platform['type'].upper(), 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw power-ups
        for power_up in game_state.get('power_ups', []):
            x, y, w, h = power_up['bbox']
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_frame, power_up['type'].upper(), 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw enemies
        for enemy in game_state.get('enemies', []):
            x, y, w, h = enemy['bbox']
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(debug_frame, enemy['type'].upper(), 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw game info
        info_text = [
            f"State: {game_state.get('game_state', 'unknown')}",
            f"Height: {game_state.get('height', 0)}",
            f"Platforms: {len(game_state.get('platforms', []))}",
            f"PowerUps: {len(game_state.get('power_ups', []))}",
            f"Enemies: {len(game_state.get('enemies', []))}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(debug_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame 