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

            # If we get landmarks
            if results.multi_hand_landmarks:
                hands = results.multi_hand_landmarks[:2]

                # Correct number of hands
                if len(hands) == 2:
                    hand1, hand2 = hands[0], hands[1]

                    

                    self.primed = hand1.landmark[2].y < hand1.landmark[8].y and hand2.landmark[2].y < hand2.landmark[8].y

                    if not self.playing:
                        if self.primed:
                            self.playing = True
                            self.firing_hand = None  
                            start = time.time()   

                    
                    if self.playing:    
                        if not self.primed and time.time() - start < 3:
                            self.playing = False
                            self.firing_hand = None
                        elif not self.primed and time.time() - start >= 3:
                            hand0_firing = hands[0].landmark[8].y <= hands[0].landmark[2].y
                            hand1_firing = hands[1].landmark[8].y <= hands[1].landmark[2].y
                            
                            if hand0_firing:
                                self.firing_hand = 'one'
                            elif hand1_firing:  
                                self.firing_hand = 'two'

                            self.playing = False
                else: 
                    self.primed = False

                for hand in hands:
                    self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Draw "1" above hand1 (first hand)
                if len(hands) > 0:
                    h, w, _ = frame.shape
                    
                    for i, hand in enumerate(hands):
                        wrist = hand.landmark[4]
                    
                        x = int(wrist.x * w)
                        y = int(wrist.y * h)
                    
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 2
                        thickness = 3
                        color = (255, 255, 255)  
                        
                        (text_width, text_height), baseline = cv2.getTextSize("1", font, font_scale, thickness)
                        
                        text_x = x - text_width // 2
                        text_y = y - 50
                        # Draw the "1"
                        cv2.putText(frame, str(i+1), (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

            else:
                self.primed = False
                self.firing_hand = None
                self.playing = False

            self.processed = frame
            time.sleep(0.01)

    def get_frame(self):
        if self.processed is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', self.processed)
        return buffer.tobytes()
    
    def reset(self):
        self.primed = False
        self.firing_hand = None
        self.playing = False

    def start_playing(self):
        self.playing = True

    def get_firing_status(self):
        """Returns which hand fired: 'left', 'right', or None"""
        return self.firing_hand, self.primed, self.playing
    
    def stop(self):
        self.running = False
        self.cap.release()
