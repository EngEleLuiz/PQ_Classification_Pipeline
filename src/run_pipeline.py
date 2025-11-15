import os
import sys
import time
import shutil
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import the feature extraction function
from feature_extraction import extract_wavelet_features

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../data/simulation_input/'))
PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, '../data/processed/'))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '../models/'))
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, '../db/'))
DB_PATH = os.path.join(DB_DIR, 'pq_events.db')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'pq_classifier.pkl')

# --- Global Model Placeholders ---
scaler = None
model = None

# --- Database Functions ---
def setup_database():
    """Creates the SQLite database and the 'events' table if they don't exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                classification TEXT NOT NULL
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"[Error] Database setup failed: {e}")
    finally:
        if conn:
            conn.close()
    print(f"Database setup complete at {DB_PATH}")

def log_to_db(filename, classification):
    """Logs a single classification event to the SQLite database."""
    timestamp = datetime.now().isoformat()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO events (timestamp, filename, classification) 
            VALUES (?, ?, ?)
        """, (timestamp, filename, classification))
        conn.commit()
    except sqlite3.Error as e:
        print(f"[Error] Failed to log to DB: {e}")
    finally:
        if conn:
            conn.close()

# --- Core Processing Function ---
def process_file(filepath):
    """
    The main ETLI (Extract, Transform, Load, Infer) function for a single file.
    """
    global scaler, model
    if not (scaler and model):
        print("[Error] Models are not loaded. Skipping file.")
        return

    try:
        print(f"[Pipeline] Processing new file: {os.path.basename(filepath)}")
        
        # 1. Extract
        df = pd.read_csv(filepath)
        signal = df['signal'].tolist()
        
        # 2. Transform
        features = extract_wavelet_features(signal)
        
        # 3. Inference
        # Reshape for a single sample
        features_reshaped = np.array(features).reshape(1, -1)
        # Scale features
        features_scaled = scaler.transform(features_reshaped)
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # 4. Log
        log_to_db(os.path.basename(filepath), str(prediction))
        
        # 5. Archive
        dest_path = os.path.join(PROCESSED_DIR, os.path.basename(filepath))
        shutil.move(filepath, dest_path)
        
        print(f"[Pipeline] Classified '{os.path.basename(filepath)}' as: {prediction}")
        print(f"[Pipeline] File moved to {PROCESSED_DIR}")

    except pd.errors.EmptyDataError:
        print(f"[Warning] File {filepath} is empty. Skipping.")
    except Exception as e:
        print(f"[Error] Failed to process {filepath}: {e}")
        # Optionally move to an 'error' folder

# --- Watchdog Event Handler ---
class FileHandler(FileSystemEventHandler):
    """
    Handles file system events for the watchdog observer.
    """
    def on_created(self, event):
        """
        Called when a file or directory is created.
        """
        if not event.is_directory and event.src_path.endswith('.csv'):
            # Wait a fraction of a second to ensure the file write is complete
            time.sleep(0.1) 
            process_file(event.src_path)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Real-Time PQ Classification Pipeline ---")
    
    # Ensure all directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Setup Database
    setup_database()
    
    # 2. Load Models
    print("Loading models...")
    try:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"[Fatal Error] Could not load model/scaler: {e}")
        print("Please run the 01-Model_Training... notebook first.")
        sys.exit(1)
    
    # 3. Start Watchdog Observer
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()
    
    print(f"[Pipeline] Monitoring folder: {INPUT_DIR}")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[Pipeline] Observer stopped. Shutting down.")
    
    observer.join()
