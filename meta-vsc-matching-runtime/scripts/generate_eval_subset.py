from pathlib import Path

import pandas as pd
from tqdm import tqdm


gt = pd.read_csv('/share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv')
query_video_dir = "/share-cvlab/yokoo/vsc/competition_data/train/query"
ref_video_dir = "/share-cvlab/yokoo/vsc/competition_data/train/reference"
noise_video_dir = "/share-cvlab/yokoo/vsc/competition_data/test/reference"
query_subset_ids = sorted(gt["query_id"].unique().tolist())
ref_subset_ids = sorted(gt["ref_id"].unique().tolist())
noise_subset_ids = [d.stem for d in sorted(Path(noise_video_dir).glob("*.mp4"))][:1024]

outdir = Path('/share-cvlab/yokoo/vsc/eval_subset')
outdir.mkdir(exist_ok=True)

for subset_ids, subset_name, subset_dir in zip(
    [query_subset_ids, ref_subset_ids, noise_subset_ids],
    ['query', 'reference', 'noise'],
    [query_video_dir, ref_video_dir, noise_video_dir],
):
    for subset_id in tqdm(subset_ids):
        video_path = Path(subset_dir) / f'{subset_id}.mp4'
        subset_outdir = outdir / subset_name
        subset_outdir.mkdir(exist_ok=True)
        out_path = subset_outdir / f'{subset_id}.mp4'
        import shutil
        shutil.copy(video_path, out_path)


gt = pd.read_csv('/share-cvlab/yokoo/vsc/competition_data/train/train_matching_ground_truth.csv')
query_video_dir = "/share-cvlab/yokoo/vsc/competition_data/train/query"
ref_video_dir = "/share-cvlab/yokoo/vsc/competition_data/train/reference"
noise_video_dir = "/share-cvlab/yokoo/vsc/competition_data/test/reference"
query_subset_ids = sorted(gt["query_id"].unique().tolist())
ref_subset_ids = sorted(gt["ref_id"].unique().tolist())
noise_subset_ids = [d.stem for d in sorted(Path(noise_video_dir).glob("*.mp4"))][:1024]

outdir = Path('/share-cvlab/yokoo/vsc/eval_subset')
outdir.mkdir(exist_ok=True)

for subset_ids, subset_name, subset_dir in zip(
    [query_subset_ids[:32], ref_subset_ids[:32], noise_subset_ids[:32]],
    ['dryrun_query', 'dryrun_reference', 'dryrun_noise'],
    [query_video_dir, ref_video_dir, noise_video_dir],
):
    for subset_id in tqdm(subset_ids):
        video_path = Path(subset_dir) / f'{subset_id}.mp4'
        subset_outdir = outdir / subset_name
        subset_outdir.mkdir(exist_ok=True)
        out_path = subset_outdir / f'{subset_id}.mp4'
        import shutil
        shutil.copy(video_path, out_path)
