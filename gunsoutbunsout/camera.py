import cv2
import threading

import mediapipe as mp
import time

class Camera:
    def __init__(self, src=1):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.processed = None
        self.running = True
        self.guns_out = False  # Track guns out status

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

            if results.multi_hand_landmarks:
                guns_out_detected = False
                for hand in results.multi_hand_landmarks:
                    self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    index_finger_tip = hand.landmark[8]
                    thumb_base = hand.landmark[2]
                    
                    if index_finger_tip.y < thumb_base.y:
                        guns_out_detected = True
                
                self.guns_out = guns_out_detected
            else:
                self.guns_out = False

            self.processed = frame
            time.sleep(0.01)

    def get_frame(self):
        if self.processed is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', self.processed)
        return buffer.tobytes()
    
    def stop(self):
        self.running = False
        self.cap.release()
