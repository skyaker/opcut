import os

INPUT_DIR = "./opcut_test_2"

OUTPUT_DIR = "./processed_series"
TEMP_DIR = "./temp_audio"

OP_DURATION = 90
SAMPLE_RATE = 4000

for folder in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(folder, exist_ok=True)
