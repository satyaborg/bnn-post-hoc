import sys
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Parameters for the training run")
parser.add_argument("--entry_point", required=True, type=str, help="The script to run")
parser.add_argument("--run_name", required=True, type=str, help="Name of the run")
parser.add_argument("--dry_run", required=True, type=int, help="0: Do all; 1: Do only first dataset; 2: Do no datasets")
parser.add_argument("--num_workers", required=False, type=int, default=4, help="Number of jobs to upload at once (default 4)")
parser.add_argument("--num_runs", required=False, type=int, default=1, help="The number of runs for each dataset (default 1)")

args, unknown = parser.parse_known_args()

entry_point = args.entry_point
run_name = args.run_name
num_workers = args.num_workers
dry_run = args.dry_run
num_runs = args.num_runs

now = datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")

prefix = "{}/{}/{}".format(year, month, day)

import tensorflow_cloud as tfc
from glob import glob
from multiprocessing.pool import ThreadPool

def run_job(dataset_name, run_idx):
    out_dir = "gs://soda-labs/regnet/models/{}/{}/{}".format(prefix, run_name, dataset_name)
    
    if dry_run == 2:
        print("Running {} on {}".format(entry_point, dataset_name))
        print("python {} --dataset_name {} --run_idx {} --seed {}".format(entry_point, dataset_name, run_idx, 42))
        return {"job_id": "{}.{}".format(dataset_name, entry_point)}

    result = tfc.run(
        entry_point=entry_point,
        requirements_txt="requirements.txt",
        distribution_strategy=None,
        entry_point_args=[
            "--dataset_name", dataset_name,
            "--run_idx", str(run_idx),
            "--seed", "42"
        ],
        job_labels={
            "dataset": dataset_name
        },
        chief_config=tfc.COMMON_MACHINE_CONFIGS["K80_1X"]
    )
    return result

def push_job(args):
    dataset_name, run_idx = args
    print("Pushing jobs for {} with run_idx {}".format(dataset_name, run_idx))
    job = run_job(dataset_name, run_idx)

    return job

datasets = [
    "abalone", "balance-scale", "credit-approval", "german", "ionosphere" ,
    "landsat-satellite", "letter", "mfeat-karhunen", "mfeat-morphological", "mfeat-zernike" ,
    "mushroom", "optdigits", "page-blocks", "segment", "spambase", "toy", "vehicle", "waveform-5000" ,
    "wdbc", "wpbc", "yeast"
]

if dry_run == 1:
    datasets = [datasets[0]]

p = ThreadPool(num_workers)

dataset_runs = []
for dataset in datasets:
    for i in range(num_runs):
        dataset_runs.append((dataset, i))

jobs = p.map(push_job, dataset_runs)

with open("job_ids.txt", "w") as f:
    for job in jobs:
        f.write(job["job_id"] + ("\n" if job != jobs[-1] else ""))
