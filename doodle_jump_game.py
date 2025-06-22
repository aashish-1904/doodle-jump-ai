#!/usr/bin/env python3
"""
Enhanced Doodle Jump Game Clone - Fixed Edition

A complete Doodle Jump implementation with proper mechanics,
no shooting, enemies that can always be avoided, and reliable platform generation.
"""

import pygame
import random
import math
import json
import time
from enum import Enum
from dataclasses import dataclass

# Initialize Pygame
pygame.init()

# Enhanced Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
FPS = 60

# Beautiful Color Palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (46, 204, 113)
DARK_GREEN = (39, 174, 96)
BLUE = (52, 152, 219)
DARK_BLUE = (41, 128, 185)
RED = (231, 76, 60)
YELLOW = (241, 196, 15)
PURPLE = (155, 89, 182)
ORANGE = (230, 126, 34)
LIGHT_GRAY = (236, 240, 241)
DARK_GRAY = (52, 73, 94)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)

# Perfect Physics
GRAVITY = 0.35
JUMP_STRENGTH = 13.0
PLATFORM_SPEED = 1.8
CHARACTER_SPEED = 7
MAX_FALL_SPEED = 15

class PlatformType(Enum):
    NORMAL = "normal"
    BREAKING = "breaking"
    MOVING = "moving"
    SPRING = "spring"
    ICE = "ice"

class PowerUpType(Enum):
    SPRING = "spring"
    ROCKET = "rocket"
    PROPELLER = "propeller"
    JETPACK = "jetpack"

class EnemyType(Enum):
    MONSTER = "monster"
    HOLE = "hole"
    UFO = "ufo"

@dataclass
class Particle:
    x: float
    y: float
    vel_x: float
    vel_y: float
    life: float
    max_life: float
    color: tuple
    size: float
    alpha: int = 255

class SimpleParticleSystem:
    def __init__(self):
        self.particles = []
        self.background_stars = []
        self._generate_background_stars()
    
    def _generate_background_stars(self):
        """Generate simple twinkling background stars"""
        for _ in range(80):
            self.background_stars.append({
                'x': random.randint(0, SCREEN_WIDTH),
                'y': random.randint(0, SCREEN_HEIGHT * 4),
                'brightness': random.uniform(0.3, 1.0),
                'twinkle_speed': random.uniform(0.02, 0.05),
                'size': random.randint(1, 3)
            })
    
    def add_explosion(self, x, y, color, count=10):
        """Simple explosion effect"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.particles.append(Particle(
                x=x, y=y,
                vel_x=math.cos(angle) * speed,
                vel_y=math.sin(angle) * speed,
                life=random.uniform(30, 60),
                max_life=60,
                color=color,
                size=random.uniform(2, 4)
            ))
    
    def add_trail(self, x, y, color, count=3):
        """Simple trail particles"""
        for _ in range(count):
            self.particles.append(Particle(
                x=x + random.uniform(-3, 3),
                y=y + random.uniform(-3, 3),
                vel_x=random.uniform(-1, 1),
                vel_y=random.uniform(-1, 1),
                life=20,
                max_life=20,
                color=color,
                size=random.uniform(1, 3)
            ))
    
    def update(self):
        # Update particles
        for particle in self.particles[:]:
            particle.x += particle.vel_x
            particle.y += particle.vel_y
            particle.vel_y += 0.1
            particle.life -= 1
            particle.alpha = max(0, int(255 * (particle.life / particle.max_life)))
            
            if particle.life <= 0:
                self.particles.remove(particle)
        
        # Update background stars
        for star in self.background_stars:
            star['brightness'] += star['twinkle_speed']
            if star['brightness'] > 1.0 or star['brightness'] < 0.3:
                star['twinkle_speed'] *= -1
    
    def draw(self, screen, camera_y):
        # Draw simple twinkling stars
        for star in self.background_stars:
            star_y = star['y'] - camera_y
            if -50 < star_y < SCREEN_HEIGHT + 50:
                alpha = int(255 * star['brightness'])
                if alpha > 0:
                    pygame.draw.circle(screen, (255, 255, 255), 
                                     (int(star['x']), int(star_y)), star['size'])
        
        # Draw particles
        for particle in self.particles:
            if particle.alpha > 0:
                particle_y = particle.y - camera_y
                if -50 < particle_y < SCREEN_HEIGHT + 50:
                    pygame.draw.circle(screen, particle.color, 
                                     (int(particle.x), int(particle_y)), int(particle.size))

class SimpleCharacter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel_x = 0
        self.vel_y = 0
        self.width = 32
        self.height = 32
        self.facing_right = True
        self.power_up = None
        self.power_up_timer = 0
        self.animation_frame = 0
    
    def update(self):
        self.vel_y += GRAVITY
        
        # Power-up effects
        if self.power_up and self.power_up_timer > 0:
            if self.power_up == PowerUpType.PROPELLER:
                self.vel_y -= 0.7
                if self.vel_y < -9:
                    self.vel_y = -9
            elif self.power_up == PowerUpType.JETPACK:
                self.vel_y -= 0.6
                if self.vel_y < -11:
                    self.vel_y = -11
        
        if self.vel_y > MAX_FALL_SPEED:
            self.vel_y = MAX_FALL_SPEED
        
        self.x += self.vel_x
        self.y += self.vel_y
        self.vel_x *= 0.86
        
        self.animation_frame = (self.animation_frame + 1) % 120
        
        if self.power_up_timer > 0:
            self.power_up_timer -= 1
            if self.power_up_timer == 0:
                self.power_up = None
        
        # Screen wrapping
        if self.x + self.width < 0:
            self.x = SCREEN_WIDTH
        elif self.x > SCREEN_WIDTH:
            self.x = -self.width
    
    def move_left(self):
        self.vel_x = max(-CHARACTER_SPEED, self.vel_x - 2.0)
        self.facing_right = False
    
    def move_right(self):
        self.vel_x = min(CHARACTER_SPEED, self.vel_x + 2.0)
        self.facing_right = True
    
    def jump(self, strength=JUMP_STRENGTH):
        self.vel_y = -strength
    
    def apply_power_up(self, power_up_type):
        """Apply power-up effects to character"""
        self.power_up = power_up_type
        if power_up_type == PowerUpType.SPRING:
            self.power_up_timer = 180
            self.jump(-18)
        elif power_up_type == PowerUpType.ROCKET:
            self.power_up_timer = 120
            self.jump(-25)
        elif power_up_type == PowerUpType.PROPELLER:
            self.power_up_timer = 300
        elif power_up_type == PowerUpType.JETPACK:
            self.power_up_timer = 240
    
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self, screen):
        # Enhanced square character with better graphics
        
        # Character shadow (offset for depth)
        shadow_rect = (int(self.x + 3), int(self.y + 3), self.width, self.height)
        pygame.draw.rect(screen, (0, 0, 0, 80), shadow_rect)
        
        # Power-up glow effect
        if self.power_up and self.power_up_timer > 0:
            glow_intensity = int(100 + 50 * math.sin(self.animation_frame * 0.1))
            glow_size = self.width + 12
            glow_rect = (int(self.x - 6), int(self.y - 6), glow_size, glow_size)
            
            # Different glow colors for different power-ups
            if self.power_up == PowerUpType.PROPELLER:
                glow_color = (0, 255, 255, glow_intensity)  # Cyan
            elif self.power_up == PowerUpType.JETPACK:
                glow_color = (255, 100, 0, glow_intensity)  # Orange
            elif self.power_up == PowerUpType.SPRING:
                glow_color = (255, 255, 0, glow_intensity)  # Yellow
            else:
                glow_color = (255, 255, 255, glow_intensity)  # White
            
            # Create glow surface
            glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, glow_color, (0, 0, glow_size, glow_size))
            screen.blit(glow_surface, (int(self.x - 6), int(self.y - 6)))
        
        # Main character body with gradient effect
        # Dark base
        pygame.draw.rect(screen, (80, 200, 80), (int(self.x), int(self.y), self.width, self.height))
        
        # Lighter center
        pygame.draw.rect(screen, (120, 255, 120), 
                        (int(self.x + 2), int(self.y + 2), self.width - 4, self.height - 4))
        
        # Highlight on top-left
        pygame.draw.rect(screen, (160, 255, 160), 
                        (int(self.x + 2), int(self.y + 2), self.width - 8, self.height - 8))
        
        # Animated eyes with blinking
        blink = (self.animation_frame % 120) < 5  # Blink every 2 seconds
        
        if not blink:
            # Normal eyes
            eye_y = int(self.y + 8)
            
            # Eye whites
            pygame.draw.circle(screen, WHITE, (int(self.x + 8), eye_y), 4)
            pygame.draw.circle(screen, WHITE, (int(self.x + 24), eye_y), 4)
            
            # Eye pupils (follow movement direction slightly)
            pupil_offset = 1 if self.facing_right else -1
            pygame.draw.circle(screen, BLACK, (int(self.x + 8 + pupil_offset), eye_y), 2)
            pygame.draw.circle(screen, BLACK, (int(self.x + 24 + pupil_offset), eye_y), 2)
            
            # Eye shine
            pygame.draw.circle(screen, WHITE, (int(self.x + 9 + pupil_offset), eye_y - 1), 1)
            pygame.draw.circle(screen, WHITE, (int(self.x + 25 + pupil_offset), eye_y - 1), 1)
        else:
            # Blinking eyes (horizontal lines)
            pygame.draw.line(screen, BLACK, (int(self.x + 5), int(self.y + 8)), 
                           (int(self.x + 11), int(self.y + 8)), 2)
            pygame.draw.line(screen, BLACK, (int(self.x + 21), int(self.y + 8)), 
                           (int(self.x + 27), int(self.y + 8)), 2)
        
        # Animated smile (changes based on velocity)
        mouth_y = int(self.y + 20)
        if self.vel_y < -5:  # Going up fast - happy
            # Big smile
            smile_points = [(int(self.x + 8), mouth_y), (int(self.x + 16), mouth_y + 4), 
                           (int(self.x + 24), mouth_y)]
            pygame.draw.lines(screen, BLACK, False, smile_points, 3)
        elif self.vel_y > 5:  # Falling fast - worried
            # Worried expression
            worry_points = [(int(self.x + 8), mouth_y + 2), (int(self.x + 16), mouth_y), 
                           (int(self.x + 24), mouth_y + 2)]
            pygame.draw.lines(screen, BLACK, False, worry_points, 2)
        else:  # Normal smile
            pygame.draw.arc(screen, BLACK, (int(self.x + 8), mouth_y - 2, 16, 8), 0, math.pi, 2)
        
        # Character outline for better definition
        pygame.draw.rect(screen, (40, 120, 40), (int(self.x), int(self.y), self.width, self.height), 2)

class SimpleEnemy:
    def __init__(self, x, y, enemy_type):
        self.x = x
        self.y = y
        self.type = enemy_type
        self.width = 24 if enemy_type == EnemyType.HOLE else 30
        self.height = 24 if enemy_type == EnemyType.HOLE else 30
        self.animation_frame = 0
        self.move_direction = 1 if random.random() > 0.5 else -1
        self.move_speed = 1.0
    
    def update(self):
        self.animation_frame += 1
        
        if self.type == EnemyType.UFO:
            # UFO movement
            self.x += self.move_direction * self.move_speed
            if self.x <= 0 or self.x >= SCREEN_WIDTH - self.width:
                self.move_direction *= -1
    
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        if self.type == EnemyType.MONSTER:
            # Simple red monster
            center_x = int(self.x + self.width//2)
            center_y = int(self.y + self.height//2)
            
            # Main body
            pygame.draw.circle(screen, (255, 50, 50), (center_x, center_y), self.width//2)
            pygame.draw.circle(screen, (255, 100, 100), (center_x, center_y), self.width//3)
            
            # Simple eyes
            pygame.draw.circle(screen, BLACK, (center_x - 6, center_y - 4), 3)
            pygame.draw.circle(screen, BLACK, (center_x + 6, center_y - 4), 3)
            pygame.draw.circle(screen, WHITE, (center_x - 6, center_y - 5), 1)
            pygame.draw.circle(screen, WHITE, (center_x + 6, center_y - 5), 1)
            
        elif self.type == EnemyType.HOLE:
            # Simple black hole
            center_x = int(self.x + self.width//2)
            center_y = int(self.y + self.height//2)
            
            # Simple rings
            pygame.draw.circle(screen, (100, 0, 150), (center_x, center_y), self.width//2 + 4, 2)
            pygame.draw.circle(screen, (60, 0, 100), (center_x, center_y), self.width//2, 3)
            pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), self.width//2 - 4)
            
        elif self.type == EnemyType.UFO:
            # Simple UFO
            center_x = int(self.x + self.width//2)
            center_y = int(self.y + self.height//2)
            
            # UFO body
            pygame.draw.ellipse(screen, (150, 150, 255), 
                              (int(self.x), center_y - self.height//4, self.width, self.height//2))
            pygame.draw.ellipse(screen, (100, 100, 200), 
                              (int(self.x + 3), center_y - self.height//4 + 2, self.width - 6, self.height//2 - 4))
            
            # Simple dome
            pygame.draw.ellipse(screen, (180, 180, 255), 
                              (int(self.x + 6), center_y - self.height//2, self.width - 12, self.height//2))

class SimplePlatform:
    def __init__(self, x, y, platform_type=PlatformType.NORMAL):
        self.x = x
        self.y = y
        self.width = 85
        self.height = 18
        self.type = platform_type
        self.broken = False
        self.move_direction = 1 if random.random() > 0.5 else -1
        self.animation_frame = 0
    
    def update(self):
        self.animation_frame += 1
        
        if self.type == PlatformType.MOVING:
            self.x += self.move_direction * PLATFORM_SPEED
            if self.x <= 0 or self.x >= SCREEN_WIDTH - self.width:
                self.move_direction *= -1
    
    def get_jump_multiplier(self):
        multipliers = {
            PlatformType.NORMAL: 1.0,
            PlatformType.BREAKING: 1.0,
            PlatformType.MOVING: 1.1,
            PlatformType.SPRING: 2.0,
            PlatformType.ICE: 0.8
        }
        return multipliers.get(self.type, 1.0)
    
    def get_color(self):
        colors = {
            PlatformType.NORMAL: GREEN,
            PlatformType.BREAKING: (139, 69, 19),
            PlatformType.MOVING: BLUE,
            PlatformType.SPRING: GOLD,
            PlatformType.ICE: (200, 220, 255)
        }
        return colors.get(self.type, GREEN)
    
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self, screen):
        if self.broken:
            return
        
        color = self.get_color()
        
        # Simple platform drawing
        pygame.draw.rect(screen, color, self.get_rect())
        pygame.draw.rect(screen, tuple(min(255, c + 40) for c in color), 
                        (self.x + 2, self.y + 2, self.width - 4, self.height - 4))
        
        # Special effects
        if self.type == PlatformType.SPRING:
            # Spring coils
            for i in range(3):
                coil_x = self.x + 15 + i * 20
                pygame.draw.circle(screen, (255, 255, 0), (coil_x, self.y + 4), 3)
        
        elif self.type == PlatformType.ICE:
            # Ice crystals
            for i in range(3):
                crystal_x = self.x + 15 + i * 20
                pygame.draw.polygon(screen, WHITE, [
                    (crystal_x, self.y + 2), 
                    (crystal_x + 2, self.y + 6),
                    (crystal_x - 2, self.y + 6)
                ])

class DoodleJumpGame:
    def __init__(self, headless=False):
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Doodle Jump - Fixed Edition")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
        else:
            self.screen = None
            self.clock = None
            self.font = None
            self.small_font = None
        
        self.headless = headless
        self.running = True
        
        # Game state
        self.character = SimpleCharacter(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        self.platforms = []
        self.enemies = []
        
        # Camera and scoring
        self.camera_y = 0
        self.score = 0
        self.game_over = False
        
        # Simple particle system
        self.particles = SimpleParticleSystem()
        
        # Load high score
        self.high_score = self.load_high_score()
        
        # Initialize platforms
        self.generate_initial_platforms()
    
    def load_high_score(self):
        try:
            with open('high_score.json', 'r') as f:
                data = json.load(f)
                return data.get('high_score', 0)
        except:
            return 0
    
    def save_high_score(self):
        try:
            with open('high_score.json', 'w') as f:
                json.dump({'high_score': self.high_score}, f)
        except:
            pass
    
    def generate_initial_platforms(self):
        """Generate initial platforms with guaranteed reachability"""
        # Starting platform
        start_platform = SimplePlatform(SCREEN_WIDTH // 2 - 42, SCREEN_HEIGHT - 50)
        self.platforms.append(start_platform)
        
        # Generate platforms going up with reliable spacing
        current_y = SCREEN_HEIGHT - 50
        for i in range(20):
            current_y -= random.randint(80, 120)  # Guaranteed reachable spacing
            
            # Random x position but ensure it's reachable
            if i == 0:
                # First platform close to start
                x = SCREEN_WIDTH // 2 - 42 + random.randint(-60, 60)
            else:
                # Subsequent platforms within jump range
                last_platform = self.platforms[-1]
                max_distance = 100  # Maximum horizontal jump distance
                x = last_platform.x + random.randint(-max_distance, max_distance)
            
            # Keep within screen bounds
            x = max(10, min(SCREEN_WIDTH - 95, x))
            
            # Choose platform type (mostly normal initially)
            platform_type = PlatformType.NORMAL
            if i > 5 and random.random() < 0.15:
                platform_type = random.choice([PlatformType.SPRING, PlatformType.MOVING])
            
            platform = SimplePlatform(x, current_y, platform_type)
            self.platforms.append(platform)
            
            # Add enemies very sparingly and strategically
            if i > 8 and random.random() < 0.08:  # Only 8% chance
                self.add_strategic_enemy(platform, current_y)
    
    def add_strategic_enemy(self, platform, platform_y):
        """Add enemies that can always be avoided"""
        enemy_type = random.choice([EnemyType.MONSTER, EnemyType.HOLE, EnemyType.UFO])
        
        if enemy_type == EnemyType.MONSTER:
            # Red monsters on platforms - can be jumped over
            enemy_x = platform.x + 10
            enemy_y = platform_y - 35
            
        elif enemy_type == EnemyType.HOLE:
            # Black holes below platforms - stay on platform to avoid
            enemy_x = platform.x + 20
            enemy_y = platform_y + 25
            
        elif enemy_type == EnemyType.UFO:
            # UFOs to the side - can be avoided by timing
            if platform.x < SCREEN_WIDTH // 2:
                enemy_x = platform.x + platform.width + 20
            else:
                enemy_x = platform.x - 50
            enemy_y = platform_y - 20
        
        # Only add if within screen bounds
        if 10 <= enemy_x <= SCREEN_WIDTH - 50:
            self.enemies.append(SimpleEnemy(enemy_x, enemy_y, enemy_type))
    
    def generate_platforms_runtime(self):
        """Generate more platforms as character climbs"""
        # Only generate if we need more platforms above
        highest_platform_y = min(p.y for p in self.platforms) if self.platforms else self.character.y
        
        if self.character.y < highest_platform_y + 400:
            return  # We have enough platforms above
        
        # Generate 8 new platforms
        current_y = highest_platform_y
        for i in range(8):
            current_y -= random.randint(80, 120)  # Guaranteed reachable spacing
            
            # Choose position within jump range of existing platforms
            nearby_platforms = [p for p in self.platforms if abs(p.y - current_y) < 150]
            if nearby_platforms:
                base_platform = random.choice(nearby_platforms)
                max_distance = 90  # Safe jump distance
                x = base_platform.x + random.randint(-max_distance, max_distance)
            else:
                x = random.randint(50, SCREEN_WIDTH - 135)
            
            # Keep within bounds
            x = max(10, min(SCREEN_WIDTH - 95, x))
            
            # Platform type
            platform_type = PlatformType.NORMAL
            if random.random() < 0.2:
                platform_type = random.choice([PlatformType.SPRING, PlatformType.MOVING, PlatformType.BREAKING])
            
            platform = SimplePlatform(x, current_y, platform_type)
            self.platforms.append(platform)
            
            # Add enemies very rarely
            if random.random() < 0.05:
                self.add_strategic_enemy(platform, current_y)

    def update(self):
        if self.game_over:
            return
        
        # Update character
        self.character.update()
        
        # Update platforms
        for platform in self.platforms:
            platform.update()
        
        # Update enemies
        for enemy in self.enemies:
            enemy.update()
        
        # Update particles
        self.particles.update()
        
        # Collisions
        self.check_platform_collisions()
        self.check_enemy_collisions()
        
        # Update camera
        self.update_camera()
        
        # Generate more platforms if needed
        self.generate_platforms_runtime()
        
        # Clean up
        self.cleanup_objects()
    
    def check_platform_collisions(self):
        """Handle character-platform collisions"""
        char_rect = self.character.get_rect()
        
        for platform in self.platforms:
            if platform.broken:
                continue
                
            platform_rect = platform.get_rect()
            
            # Collision detection
            if (char_rect.colliderect(platform_rect) and 
                self.character.vel_y > 0 and 
                self.character.y < platform.y):
                
                self.character.y = platform.y - self.character.height
                jump_strength = JUMP_STRENGTH * platform.get_jump_multiplier()
                self.character.jump(jump_strength)
                
                # Particle effects
                if platform.type == PlatformType.SPRING:
                    self.particles.add_explosion(
                        self.character.x + self.character.width//2,
                        platform.y, GOLD, 15
                    )
                else:
                    self.particles.add_trail(
                        self.character.x + self.character.width//2,
                        platform.y, WHITE, 5
                    )
                
                if platform.type == PlatformType.BREAKING:
                    platform.broken = True
                    self.particles.add_explosion(
                        platform.x + platform.width//2,
                        platform.y, (139, 69, 19), 10
                    )
                break
    
    def check_enemy_collisions(self):
        """Handle character-enemy collisions"""
        char_rect = self.character.get_rect()
        
        for enemy in self.enemies:
            if char_rect.colliderect(enemy.get_rect()):
                self.particles.add_explosion(
                    self.character.x + self.character.width//2,
                    self.character.y + self.character.height//2,
                    RED, 20
                )
                self.game_over = True
                if self.score > self.high_score:
                    self.high_score = self.score
                    self.save_high_score()
                return
    
    def update_camera(self):
        """Update camera to follow character"""
        target_camera_y = self.character.y - SCREEN_HEIGHT * 0.7
        if target_camera_y < self.camera_y:
            self.camera_y += (target_camera_y - self.camera_y) * 0.1
        
        # Update score based on height
        height = max(0, -self.character.y + SCREEN_HEIGHT - 100)
        self.score = max(self.score, int(height / 4))
        
        # Check if character fell
        if self.character.y > self.camera_y + SCREEN_HEIGHT + 100:
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()

    def cleanup_objects(self):
        """Clean up off-screen objects more aggressively for better performance"""
        # More aggressive cleanup to prevent slowdown
        # Remove platforms that are far below (increased range)
        self.platforms = [p for p in self.platforms if p.y > self.camera_y - 300]
        
        # Remove enemies that are far below (increased range)
        self.enemies = [e for e in self.enemies if e.y > self.camera_y - 300]
        
        # Limit total number of platforms to prevent memory buildup
        if len(self.platforms) > 30:
            # Keep only the most recent platforms
            self.platforms = self.platforms[-25:]
        
        # Limit total number of enemies
        if len(self.enemies) > 15:
            # Keep only the most recent enemies
            self.enemies = self.enemies[-10:]

    def draw(self):
        if self.headless:
            return
        
        # Simple gradient background
        for y in range(SCREEN_HEIGHT):
            intensity = int(50 + (y / SCREEN_HEIGHT) * 100)
            color = (intensity//3, intensity//2, intensity)
            pygame.draw.line(self.screen, color, (0, y), (SCREEN_WIDTH, y))
        
        # Draw particles (stars)
        self.particles.draw(self.screen, self.camera_y)
        
        # Only draw objects that are actually visible (performance optimization)
        visible_platforms = [p for p in self.platforms 
                           if not p.broken and 
                           self.camera_y - 50 < p.y < self.camera_y + SCREEN_HEIGHT + 50]
        
        visible_enemies = [e for e in self.enemies 
                         if self.camera_y - 50 < e.y < self.camera_y + SCREEN_HEIGHT + 50]
        
        # Draw platforms
        for platform in visible_platforms:
            # Create a copy with adjusted position for drawing
            draw_platform = SimplePlatform(platform.x, platform.y - self.camera_y, platform.type)
            draw_platform.broken = platform.broken
            draw_platform.animation_frame = platform.animation_frame
            draw_platform.draw(self.screen)
        
        # Draw enemies
        for enemy in visible_enemies:
            # Create a copy with adjusted position for drawing
            draw_enemy = SimpleEnemy(enemy.x, enemy.y - self.camera_y, enemy.type)
            draw_enemy.animation_frame = enemy.animation_frame
            draw_enemy.move_direction = enemy.move_direction
            draw_enemy.draw(self.screen)
        
        # Draw character
        draw_character = SimpleCharacter(self.character.x, self.character.y - self.camera_y)
        draw_character.facing_right = self.character.facing_right
        draw_character.animation_frame = self.character.animation_frame
        draw_character.power_up = self.character.power_up
        draw_character.power_up_timer = self.character.power_up_timer
        draw_character.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_ui(self):
        # Simple score display
        score_text = f"Score: {self.score:,}"
        score_surface = self.font.render(score_text, True, WHITE)
        self.screen.blit(score_surface, (20, 20))
        
        # High score
        high_score_text = f"Best: {self.high_score:,}"
        high_score_surface = self.small_font.render(high_score_text, True, GOLD)
        self.screen.blit(high_score_surface, (20, 60))
        
        # Controls
        controls_text = "← → Move"
        controls_surface = self.small_font.render(controls_text, True, WHITE)
        self.screen.blit(controls_surface, (20, 90))
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            # Game over text
            game_over_text = "GAME OVER"
            game_over_surface = pygame.font.Font(None, 72).render(game_over_text, True, WHITE)
            game_over_rect = game_over_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
            self.screen.blit(game_over_surface, game_over_rect)
            
            # Final score
            final_score_text = f"Final Score: {self.score:,}"
            final_score_surface = self.font.render(final_score_text, True, WHITE)
            final_score_rect = final_score_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(final_score_surface, final_score_rect)
            
            # High score celebration
            if self.score == self.high_score and self.score > 0:
                new_high_text = "NEW HIGH SCORE!"
                new_high_surface = self.font.render(new_high_text, True, GOLD)
                new_high_rect = new_high_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40))
                self.screen.blit(new_high_surface, new_high_rect)
            
            # Restart instruction
            restart_text = "Press R to Restart"
            restart_surface = self.font.render(restart_text, True, WHITE)
            restart_rect = restart_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80))
            self.screen.blit(restart_surface, restart_rect)

    def handle_input(self, keys_pressed):
        """Handle player input - NO SHOOTING"""
        if keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
            self.character.move_left()
        if keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
            self.character.move_right()

    def reset(self):
        self.character = SimpleCharacter(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        self.platforms.clear()
        self.enemies.clear()
        self.particles = SimpleParticleSystem()
        self.camera_y = 0
        self.score = 0
        self.game_over = False
        self.generate_initial_platforms()
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset()
            
            if not self.game_over:
                keys = pygame.key.get_pressed()
                self.handle_input(keys)
            
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = DoodleJumpGame()
    game.run() 