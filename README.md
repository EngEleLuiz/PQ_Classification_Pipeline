# ⚡ Real-Time Power Quality (PQ) Classification Pipeline
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
```markdown
This project implements a real-time, event-driven pipeline to monitor, process, and classify power quality (PQ) disturbances from waveform data.

It uses the `watchdog` library to monitor a directory for new signal files (e.g., from a simulation or data logger), applies a Discrete Wavelet Transform (DWT) for feature extraction, and uses a pre-trained `scikit-learn` model to classify events like Sag, Swell, Harmonics, and Normal. All classification results are logged in real-time to an SQLite database.

## 📊 Project Workflow

The pipeline operates on a simple, event-driven producer-consumer model:

1.  Produce: A new waveform file (e.g., `.csv` or `.parquet`) is dropped into the `data/simulation_input/` directory.
2.  Detect: The `watchdog` file system monitor (running in `run_pipeline.py`) immediately detects the new file.
3.  Extract: The system reads the file, loads the signal, and applies DWT-based feature engineering (using `feature_extraction.py`) to generate a feature vector (e.g., energy, entropy, standard deviation of wavelet coefficients).
4.  Classify: The feature vector is fed into the pre-trained model loaded from the `models/` directory.
5.  Log: The model's prediction (e.g., 'Sag', 'Swell'), filename, and timestamp are saved to the `db/pq_events.db` SQLite database.



## ✨ Key Features

 Real-Time Monitoring: Event-driven architecture using `watchdog` for low-latency processing.
 Advanced Signal Processing: Leverages the Discrete Wavelet Transform (DWT) via `pywt` for robust feature engineering that is well-suited for non-stationary signals.
 ML Classification: Uses a `scikit-learn` model for accurate disturbance identification.
 Persistent Logging: Logs all events to an `SQLite` database for auditing and future analysis.
 Data Simulation: Includes a `simulator.py` to generate synthetic test data and test the pipeline's responsiveness.
 Decoupled & Scalable: Clean separation between training (`notebooks/`), feature extraction (`src/feature_extraction.py`), and the pipeline service (`src/run_pipeline.py`).

## 🛠️ Tech Stack

 Python 3.x
 Core Libraries:
     `pandas`: Data manipulation and file I/O.
     `scikit-learn`: Machine learning (model training and prediction).
     `pywt`: Discrete Wavelet Transform for signal processing.
     `scipy`: Scientific and signal processing utilities.
     `numpy`: Numerical operations.
 Pipeline & Storage:
     `watchdog`: File system monitoring.
     `joblib`: Persisting (saving/loading) the trained ML model.
     `sqlite3`: (Python built-in) for database logging.
     `pyarrow`: (Optional, but recommended) for efficient `.parquet` file handling.
 Experimentation:
     `jupyterlab` / `notebooks`: Model development and feature engineering.
     `matplotlib` / `seaborn`: Data visualization.

---

## 📂 Project Structure

```

PQ\_Classification\_Pipeline/
├── data/
│   ├── raw/                \# Raw data for model training (e.g., IEEE datasets)
│   ├── simulation\_input/   \# Monitored folder: Pipeline watches this
│   └── processed/          \# Processed data from training notebook
├── notebooks/
│   └── 01-Model\_Training\_and\_Feature\_Engineering.ipynb \# Train & save model
├── src/
│   ├── feature\_extraction.py   \# DWT feature engineering functions
│   ├── run\_pipeline.py         \# Main pipeline script (runs watchdog)
│   └── simulator.py            \# Generates test files for the pipeline
├── models/                     \# Stores the trained model (e.g., model.joblib)
├── db/                         \# Stores the SQLite database (e.g., pq\_events.db)
├── .gitignore
├── README.md
└── requirements.txt

````

---

## 🚀 Getting Started

### 1. Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/PQ_Classification_Pipeline.git](https://github.com/YourUsername/PQ_Classification_Pipeline.git)
    cd PQ_Classification_Pipeline
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Model Training (Prerequisite)

The real-time pipeline requires a pre-trained model to function.

1.  Add your data: Place your raw PQ signal data (e.g., `.csv` files) into the `data/raw/` folder. This data is used only for training the model.
2.  Run the notebook: Open and run the `notebooks/01-Model_Training_and_Feature_Engineering.ipynb` notebook.
3.  Verify: This notebook will process the raw data, extract DWT features, train a classifier (e.g., Random Forest), and save the final model artifact (e.g., `model.joblib`) to the `models/` directory.

### 3. Running the Pipeline

You will need two separate terminal sessions running concurrently.

#### Terminal 1: Run the Main Pipeline (Listener)

This script will first connect to (or create) the `SQLite` database, load the model from `models/`, and then begin monitoring the `data/simulation_input/` folder.

```bash
python src/run_pipeline.py
````

You should see an output message like: `[Pipeline] Monitoring started. Watching folder: /path/to/PQ_Classification_Pipeline/data/simulation_input`

#### Terminal 2: Run the Simulator (Producer)

This script will generate synthetic waveform data (e.g., a signal with a "Sag") and save it as a new file in the `data/simulation_input/` folder.

```bash
python src/simulator.py
```

### 4\. Check the Results

   In Terminal 1 (Pipeline): As soon as the simulator drops a file, you will see real-time logs as the pipeline processes and classifies it.
    ```
    [Pipeline] Detected new file: sample_12345.csv
    [Pipeline] Processing signal...
    [Pipeline] Features extracted. Classifying...
    [Pipeline] Classification: 'Sag' (Confidence: 0.92)
    [Pipeline] Result logged to database.
    ```
   In the Database: You can use a tool like DB Browser for SQLite to open the `db/pq_events.db` file and see the new records being added to the `classifications` table.


```
```
## 👨‍💻 Author

**[Luiz Rosa]** *Power Systems Engineer | Python Developer* [[LinkedIn](https://www.linkedin.com/in/luiz-gustavo-rosa-12407536b/)]
