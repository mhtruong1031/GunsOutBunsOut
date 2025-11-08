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

        self.primed = False
        self.firing_status: list[bool] = [False, False]  # Track guns out status
        self.fired_hand = None  # Track which hand fired: 'left' or 'right'

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
                hands = results.multi_hand_landmarks[:2]
                middle_x = frame.shape[1] // 2

                if len(hands) == 2:
                    hand1, hand2 = hands[0], hands[1]

                    if hand1.landmark[2].y < hand1.landmark[8].y and hand2.landmark[2].y < hand2.landmark[8].y:
                        self.primed = True
                    else:
                        self.primed = False
                        self.firing_status = [False, False]
                        self.fired_hand = None
                    
                    # Check if index finger (landmark 8) is lifted above landmark 2
                    # First hand that fires sets its firing_status to True
                    if self.primed:
                        hand0_firing = hands[0].landmark[8].y < hands[0].landmark[2].y
                        hand1_firing = hands[1].landmark[8].y < hands[1].landmark[2].y
                        
                        # Reset if neither hand is firing (allows for new round)
                        '''
                        if not hand0_firing and not hand1_firing:
                            self.firing_status = [False, False]
                            self.fired_hand = None
                        # Check which hand fired first (only set if not already fired)
                        elif hand0_firing and not self.firing_status[0] and not self.firing_status[1]:
                            self.firing_status[0] = True
                            # Determine which side this hand is on
                            self.fired_hand = 'left' if hand0_is_left else 'right'
                        elif hand1_firing and not self.firing_status[0] and not self.firing_status[1]:
                            self.firing_status[1] = True
                            # Determine which side this hand is on
                            self.fired_hand = 'left' if hand1_is_left else 'right'
                        '''

                else:
                    self.primed = False
                    self.firing_status = [False, False]

                for hand in hands:
                    self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            else:
                self.primed = False
                self.firing_status = [False, False]
                self.fired_hand = None

            self.processed = frame
            time.sleep(0.01)

    def get_frame(self):
        if self.processed is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', self.processed)
        return buffer.tobytes()
    
    def get_firing_status(self):
        """Returns which hand fired: 'left', 'right', or None"""
        return self.fired_hand, self.primed
    
    def stop(self):
        self.running = False
        self.cap.release()
