import cv2
import threading
import math

import mediapipe as mp
import time

class Bullet():
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def iterate(self):
        self.x += self.dx
        self.y += self.dy

class Camera:
    def __init__(self, src=2):
        # Camera handling
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.processed = None
        self.running = True

        # Game state
        self.playing = False
        self.primed = False
        self.primed_start_time = None
        self.winning_side = None
        
        # HP tracking
        self.left_hp = 3
        self.right_hp = 3
        
        # Bullet system
        self.bullets = []
        self.bullet_speed = 15  # pixels per frame
        self.prev_firing_states = {}  # Track previous firing states to detect transitions
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands.Hands()
        self.drawer = mp.solutions.drawing_utils

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _distance(self, x1, y1, x2, y2):
        """Calculate distance between two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            h, w, _ = frame.shape
            center_x = w // 2
            
            # Draw center line
            cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 2)
            
            # Reset game if no hands detected
            if not results.multi_hand_landmarks:
                if self.playing:
                    self._reset_game()
                self.primed = False
                self.primed_start_time = None
            else:
                hands = results.multi_hand_landmarks[:2]
                
                if len(hands) == 2:
                    hand1, hand2 = hands[0], hands[1]
                    
                    hand1_x = hand1.landmark[0].x * w
                    hand2_x = hand2.landmark[0].x * w
                    hand1_side = 'left' if hand1_x < center_x else 'right'
                    hand2_side = 'left' if hand2_x < center_x else 'right'
                    
                    # Check if hands are on opposite sides and primed (both index fingers up)
                    hand1_finger_up = hand1.landmark[8].y >= hand1.landmark[2].y
                    hand2_finger_up = hand2.landmark[8].y >= hand2.landmark[2].y
                    on_opposite_sides = hand1_side != hand2_side
                    
                    current_primed = hand1_finger_up and hand2_finger_up and on_opposite_sides
                    
                    # Track priming time - start timer when primed, reset when not primed
                    if current_primed:
                        if self.primed_start_time is None:
                            self.primed_start_time = time.time()
                        elif not self.playing:
                            if self.winning_side is not None:
                                if time.time() - self.primed_start_time >= 0.75:
                                    # Reset game and start new round
                                    self._reset_game()
                                    self._start_game()
                            else:
                                if time.time() - self.primed_start_time >= 3.0:
                                    self._start_game()
                    else:
                        self.primed_start_time = None
                    
                    self.primed = current_primed
                    
                    # Gameplay logic
                    if self.playing and self.winning_side is None:
                        # Track hand positions and states for bullet creation and hit detection
                        hand_info = {}
                        for i, hand in enumerate(hands):
                            hand_x = hand.landmark[0].x * w
                            side = 'left' if hand_x < center_x else 'right'
                            
                            # Only keep one hand per side (prefer the one we haven't seen yet, or replace if closer to center)
                            if side not in hand_info:
                                index_finger_tip_y = hand.landmark[8].y * h
                                index_finger_mcp_y = hand.landmark[2].y * h
                                is_firing = index_finger_tip_y > index_finger_mcp_y
                                
                                # Get index finger tip position for bullet spawn
                                index_tip_x = hand.landmark[8].x * w
                                index_tip_y = hand.landmark[8].y * h
                                
                                # Get landmark 9 (pinky MCP) for hit detection
                                pinky_mcp_x = hand.landmark[9].x * w
                                pinky_mcp_y = hand.landmark[9].y * h
                                
                                # Get wrist (landmark 0) for hit detection radius
                                wrist_x = hand.landmark[0].x * w
                                wrist_y = hand.landmark[0].y * h
                                
                                # Calculate hit detection radius
                                hit_radius = self._distance(pinky_mcp_x, pinky_mcp_y, wrist_x, wrist_y)
                                
                                hand_info[side] = {
                                    'hand': hand,
                                    'side': side,
                                    'is_firing': is_firing,
                                    'index_tip_x': index_tip_x,
                                    'index_tip_y': index_tip_y,
                                    'pinky_mcp_x': pinky_mcp_x,
                                    'pinky_mcp_y': pinky_mcp_y,
                                    'hit_radius': hit_radius,
                                    'hand_x': hand_x
                                }
                        
                        hand_info_list = list(hand_info.values())
                        
                        if self.winning_side is None:
                            for info in hand_info_list:
                                hand_id = info['side']
                                prev_firing = self.prev_firing_states.get(hand_id, False)
                                
                                # Create bullet when transitioning from not firing to firing
                                if info['is_firing'] and not prev_firing:
                                    if info['side'] == 'left':
                                        bullet_dx = self.bullet_speed  # Moving right
                                    else:
                                        bullet_dx = -self.bullet_speed  # Moving left
                                    
                                    bullet = Bullet(
                                        info['index_tip_x'],
                                        info['index_tip_y'],
                                        bullet_dx,
                                        0 
                                    )
                                    self.bullets.append({
                                        'bullet': bullet,
                                        'owner_side': info['side']
                                    })
                                
                                self.prev_firing_states[hand_id] = info['is_firing']
                        else:
                            self.prev_firing_states = {}
                        
                        # Update bullet positions and check collisions
                        bullets_to_remove = []
                        game_ended = False
                        
                        for bullet_data in self.bullets[:]: 
                            if game_ended:
                                break
                                
                            bullet = bullet_data['bullet']
                            owner_side = bullet_data['owner_side']

                            bullet.iterate() # Move bullet
                            if bullet.x < 0 or bullet.x > w or bullet.y < 0 or bullet.y > h:
                                bullets_to_remove.append(bullet_data)
                                continue
                            
                            # Check collision with enemy hands
                            for info in hand_info_list:
                                if info['side'] == owner_side:
                                    continue
                                
                                dist = self._distance(
                                    bullet.x, bullet.y,
                                    info['pinky_mcp_x'], info['pinky_mcp_y']
                                )
                                
                                # Check if bullet is within hit radius
                                if dist <= info['hit_radius']:
                                    if info['side'] == 'left':
                                        self.left_hp = max(0, self.left_hp - 1)
                                    else:
                                        self.right_hp = max(0, self.right_hp - 1)
                                    
                                    bullets_to_remove.append(bullet_data)
                                    
                                    # Check for game over
                                    if self.left_hp <= 0:
                                        self.winning_side = 'right'
                                        self.playing = False
                                        game_ended = True
                                        self.primed_start_time = None  
                                    elif self.right_hp <= 0:
                                        self.winning_side = 'left'
                                        self.playing = False
                                        game_ended = True
                                        self.primed_start_time = None  
                                    break
                        
                        # Remove bullets that hit or went out of bounds
                        for bullet_data in bullets_to_remove:
                            if bullet_data in self.bullets:
                                self.bullets.remove(bullet_data)
                        
                        if game_ended:
                            self.bullets = []
                        
                        for bullet_data in self.bullets:
                            bullet = bullet_data['bullet']
                            cv2.circle(frame, (int(bullet.x), int(bullet.y)), 10, (255, 255, 255), -1)
                    
                    # Draw hand landmarks
                    for hand in hands:
                        self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Draw HP
            if self.playing or self.winning_side is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                left_hp_text = f"HP: {self.left_hp}/3"
                (text_width, text_height), baseline = cv2.getTextSize(left_hp_text, font, font_scale, thickness)
                cv2.putText(frame, left_hp_text, (20, 50), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                right_hp_text = f"HP: {self.right_hp}/3"
                (text_width, text_height), baseline = cv2.getTextSize(right_hp_text, font, font_scale, thickness)
                cv2.putText(frame, right_hp_text, (w - text_width - 20, 50), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            
            # Draw win/lose overlay
            if self.winning_side is not None:
                overlay = frame.copy()
                
                if self.winning_side == 'left':
                    cv2.rectangle(overlay, (0, 0), (center_x, h), (0, 255, 0), -1)  # Green
                    cv2.rectangle(overlay, (center_x, 0), (w, h), (0, 0, 255), -1)  # Red
                else:
                    cv2.rectangle(overlay, (center_x, 0), (w, h), (0, 255, 0), -1)  # Green
                    cv2.rectangle(overlay, (0, 0), (center_x, h), (0, 0, 255), -1)  # Red
                
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            self.processed = frame
            time.sleep(0.01)

    def _start_game(self):
        """Initialize game state"""
        self.playing = True
        self.left_hp = 3
        self.right_hp = 3
        self.bullets = []
        self.winning_side = None
        self.prev_firing_states = {}
        self.primed_start_time = None  # Reset priming timer

    def _reset_game(self):
        """Reset game state"""
        self.playing = False
        self.left_hp = 3
        self.right_hp = 3
        self.bullets = []
        self.winning_side = None
        self.prev_firing_states = {}

    def get_frame(self):
        if self.processed is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', self.processed)
        return buffer.tobytes()
    
    def reset(self):
        """Public method to reset the game"""
        self._reset_game()
        self.primed = False
        self.primed_start_time = None

    def get_firing_status(self):
        """Returns game status: firing_hand (deprecated), playing, and HP info"""
        return None, self.playing, {
            'playing': self.playing,
            'primed': self.primed,
            'left_hp': self.left_hp,
            'right_hp': self.right_hp,
            'winning_side': self.winning_side
        }
    
    def stop(self):
        self.running = False
        self.cap.release()
