# Functions for reading indexing and organzing demos


import json
import h5py
import pandas as pd




def get_task_instructions(index, suite="libero_90"):
    # Select tasks from the requested dataset
    tasks = index[index["suite"] == suite]

    # Each task has 50 demonstrations but only one instruction
    return tasks[["task", "instruction"]].drop_duplicates()



def build_dataset_index(dataset_root):
    records = []

    # Process both LIBERO datasets
    for suite in ["libero_90", "libero_10"]:
        files = sorted((dataset_root / suite).glob("*.hdf5"))

        for file in files:
            with h5py.File(file, "r") as f:
                data = f["data"]

                # Extract the task instruction
                info = json.loads(data.attrs["problem_info"])
                instruction = info["language_instruction"]

                # Index each demonstration
                # Sort demonstration IDs numerically
                demo_ids = [key for key in data if key.startswith("demo_")]
                demo_ids.sort(key=lambda x: int(x.split("_")[1]))

                for demo_id in demo_ids:
                    if not demo_id.startswith("demo_"):
                        continue

                    records.append({
                        "suite": suite,
                        "task": file.stem,
                        "instruction": instruction,
                        "demo_id": demo_id,
                        "file_path": str(file),
                        "length": len(data[demo_id]["actions"])
                    })

    return pd.DataFrame(records)