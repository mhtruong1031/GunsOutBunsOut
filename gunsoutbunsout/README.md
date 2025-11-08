# Guns Out Buns Out - Flask App

A Flask web application.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

The app will be available at `http://localhost:5000`

## Project Structure

```
gunsoutbunsout/
├── app/
│   ├── __init__.py      # Flask app factory
│   ├── routes.py        # Route handlers
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── main.js
│   └── templates/
│       └── index.html
├── config.py            # Configuration
├── run.py               # Application entry point
├── requirements.txt     # Python dependencies
└── README.md
```

## Environment Variables

- `SECRET_KEY`: Secret key for Flask sessions (defaults to dev key)
- `FLASK_DEBUG`: Set to 'True' to enable debug mode

