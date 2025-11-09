import cv2
import threading

import mediapipe as mp
import time

class Camera:
    def __init__(self, src=2):
        # Camera handling
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.processed = None
        self.running = True

        self.playing = False
        self.primed = False
        self.firing_hand = None
        self.primed_start_time = None
        self.play_start_time = None
        self.winning_side = None  

        self.mp_hands = mp.solutions.hands.Hands()
        self.drawer = mp.solutions.drawing_utils

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            # Center line type shi
            h, w, _ = frame.shape
            center_x = w // 2
            cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 2)
            
            # If we get landmarks
            if results.multi_hand_landmarks:
                hands = results.multi_hand_landmarks[:2]
    
                if len(hands) == 2:
                    hand1, hand2 = hands[0], hands[1]

                    # Check if hands are on opposite sides of the screen and primed
                    fingers_up = hand1.landmark[2].y < hand1.landmark[8].y and hand2.landmark[2].y < hand2.landmark[8].y
                    hand1_x = hand1.landmark[0].x * w  
                    hand2_x = hand2.landmark[0].x * w
                    on_opposite_sides = (hand1_x < center_x and hand2_x > center_x) or (hand1_x > center_x and hand2_x < center_x)

                    current_primed = fingers_up and on_opposite_sides
                    
                    # Update primed times
                    if current_primed and not self.primed:
                        self.primed_start_time = time.time()
                        self.winning_side = None
                    elif not current_primed:
                        self.primed_start_time = None
                    
                    self.primed = current_primed

                    if not self.playing:
                        # Only start playing if primed for at least 0.75 seconds
                        if self.primed and self.primed_start_time is not None:
                            if time.time() - self.primed_start_time >= 0.75:
                                self.playing = True
                                self.firing_hand = None  
                                self.play_start_time = time.time()   

                    
                    if self.playing:    
                        if not self.primed and self.play_start_time is not None and time.time() - self.play_start_time < 3:
                            self.playing = False
                            self.firing_hand = None
                            self.play_start_time = None
                        elif not self.primed and self.play_start_time is not None and time.time() - self.play_start_time >= 3:
                            hand0_firing = hands[0].landmark[8].y <= hands[0].landmark[2].y
                            hand1_firing = hands[1].landmark[8].y <= hands[1].landmark[2].y
                            
                            if hand0_firing:
                                self.firing_hand = 'one'
                                hand0_x = hands[0].landmark[0].x * w
                                self.winning_side = 'left' if hand0_x < center_x else 'right'
                            elif hand1_firing:  
                                self.firing_hand = 'two'
                                hand1_x = hands[1].landmark[0].x * w
                                self.winning_side = 'left' if hand1_x < center_x else 'right'

                            self.playing = False
                            self.play_start_time = None
                else: 
                    self.primed = False
                    self.primed_start_time = None
                    self.playing = False
                    self.play_start_time = None

                for hand in hands:
                    self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Draw hand numbers above each hand
                if len(hands) > 0:
                    for i, hand in enumerate(hands):
                        wrist = hand.landmark[4]
                    
                        x = int(wrist.x * w)
                        y = int(wrist.y * h)
                    
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 2
                        thickness = 3
                        color = (255, 255, 255)  
                        
                        (text_width, text_height), baseline = cv2.getTextSize(str(i+1), font, font_scale, thickness)
                        
                        text_x = x - text_width // 2
                        text_y = y - 50
                        cv2.putText(frame, str(i+1), (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

            else:
                self.primed = False
                self.primed_start_time = None
                self.firing_hand = None
                self.playing = False
                self.play_start_time = None

            # Draw colored overlays for winner/loser if there's a winning side
            if self.winning_side is not None:
                overlay = frame.copy()
                
                if self.winning_side == 'left':
                    # Left side wins (green), right side loses (red)
                    cv2.rectangle(overlay, (0, 0), (center_x, h), (0, 255, 0), -1)  # Green
                    cv2.rectangle(overlay, (center_x, 0), (w, h), (0, 0, 255), -1)  # Red
                else: 
                    # Right side wins (green), left side loses (red)
                    cv2.rectangle(overlay, (center_x, 0), (w, h), (0, 255, 0), -1)  # Green
                    cv2.rectangle(overlay, (0, 0), (center_x, h), (0, 0, 255), -1)  # Red
                
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            self.processed = frame
            time.sleep(0.01)

    def get_frame(self):
        if self.processed is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', self.processed)
        return buffer.tobytes()
    
    def reset(self):
        self.primed = False
        self.primed_start_time = None
        self.firing_hand = None
        self.playing = False
        self.play_start_time = None
        self.winning_side = None

    def start_playing(self):
        self.playing = True

    def get_firing_status(self):
        """Returns which hand fired: 'left', 'right', or None"""
        return self.firing_hand, self.playing, self.playing
    
    def stop(self):
        self.running = False
        self.cap.release()
