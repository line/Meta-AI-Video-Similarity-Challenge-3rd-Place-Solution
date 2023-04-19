"""
Copyright 2023 LINE Corporation

LINE Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import time
from shutil import copyfile

ROOT_DIRECTORY = Path("/code_execution")
DATA_DIRECTORY = Path("/data")
QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY / "query"
OUTPUT_FILE = ROOT_DIRECTORY / "subset_query_descriptors.npz"
QUERY_SUBSET_FILE = DATA_DIRECTORY / "query_subset.csv"


def main():

    num_videos = len(pd.read_csv(QUERY_SUBSET_FILE))
    time_deadline_sec = num_videos * (10 + 1)  # 10 sec per video + 1 sec for overhead
    start_time = time.time()

    cmd = f"""
    conda run --no-capture-output -n condaenv python -m src.inference \
        --accelerator=cuda --processes=1 --fps 2 --stride 3 \
        --dataset_path {str(QRY_VIDEOS_DIRECTORY)} \
        --output_file {str(OUTPUT_FILE)} \
        --read_type tensor \
        --video_reader DECORD
    """
    subprocess.run(cmd.split())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} sec. (deadline: {time_deadline_sec} sec)")

    if elapsed_time > time_deadline_sec:
        print("Time limit exceeded.")
    else:
        print("Time limit not exceeded.")


if __name__ == "__main__":
    main()