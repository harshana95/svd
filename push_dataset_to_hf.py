import os
import shutil
from os.path import join

from datasets import disable_caching, load_dataset
from huggingface_hub import HfApi
ds_dir = "/scratch/gilbreth/wweligam/dataset/test_image_dataset/synthetic_hybrid_Flickr2k_gt_v2_gaussian_synthetic_PCA"
md_dir = join(ds_dir, 'metadata')
dataset_name = os.path.basename(ds_dir)

disable_caching()

shutil.copyfile('./dataset/loading_script.py', join(ds_dir, f'{dataset_name}.py'))
shutil.rmtree('./.cachehf', ignore_errors=True)
dataset = load_dataset(ds_dir, trust_remote_code=True, cache_dir='./.cachehf')
shutil.rmtree('./.cachehf', ignore_errors=True)
print(f"Length of the created dataset {len(dataset)}")

repoid = f"harshana95/{dataset_name}"
dataset.push_to_hub(repoid, num_shards={'train': 100, 'val': 1})

api = HfApi()
api.upload_folder(
    folder_path=md_dir,
    repo_id=repoid,
    path_in_repo="metadata",
    repo_type="dataset",

)

