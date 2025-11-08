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

@app.route('/play')
def play():
    return render_template('play.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    fired_hand, primed, playing = camera.get_firing_status()
    print(fired_hand, primed, playing)
    return jsonify({
        'guns_out': fired_hand is not None,
        'fired_hand': fired_hand,
        'primed': primed
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
