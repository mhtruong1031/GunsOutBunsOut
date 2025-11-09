from flask import Flask, render_template, Response, jsonify
from camera import Camera

app = Flask(__name__)
camera = Camera()

def gen(cam):
    while True:
        frame = cam.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play')
def play():
    return render_template('play.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    fired_hand, playing, status_info = camera.get_firing_status()
    return jsonify({
        'playing': status_info.get('playing', False),
        'primed': status_info.get('primed', False),
        'left_hp': status_info.get('left_hp', 3),
        'right_hp': status_info.get('right_hp', 3),
        'winning_side': status_info.get('winning_side'),
        'game_over': status_info.get('winning_side') is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
