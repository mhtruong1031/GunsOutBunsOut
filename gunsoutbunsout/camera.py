import cv2

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # run your CV model here
        # frame = your_model.process(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
