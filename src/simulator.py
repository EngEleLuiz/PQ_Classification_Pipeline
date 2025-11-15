import os
import glob
import time
import random
import shutil

RAW_DATA_PATH = r"C:\Users\luizg\Downloads\ProjetoGithub\PQ_Classification_Pipeline\synthetic_signals"
SIMULATION_INPUT_PATH = r'data/simulation_input/'


def run_simulator():
    print("--- PQ Event Simulator ---")
    os.makedirs(SIMULATION_INPUT_PATH, exist_ok=True)

    all_raw_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))

    if not all_raw_files:
        print(f"Error: No raw data files found in '{RAW_DATA_PATH}'.")
        return

    print(f"Found {len(all_raw_files)} CSV files to simulate.")
    print(f"Dropping new files into: {os.path.abspath(SIMULATION_INPUT_PATH)}")

    try:
        while True:
            source_file = random.choice(all_raw_files)


            timestamp = int(time.time() * 1000)
            base_name = os.path.basename(source_file)
            dest_name = f"{timestamp}_{base_name}"
            dest_file = os.path.join(SIMULATION_INPUT_PATH, dest_name)

            shutil.copy(source_file, dest_file)
            print(f"[Simulator] Uploaded event: {dest_name}")

            time.sleep(random.uniform(2, 5))

    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    run_simulator()