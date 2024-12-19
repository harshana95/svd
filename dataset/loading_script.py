import json
import os
import datasets

class ImagePairDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="An image pair dataset",
            features=datasets.Features(
                {
                    "gt": datasets.Image(),
                    "blur": datasets.Image(),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        fpath = os.path.realpath(__file__)
        js = json.loads(open(str(fpath)[:-3]+'.json').read())
        data_dir = os.path.dirname(js['original file path'])
        print(data_dir, os.getcwd(), os.path.realpath(__file__))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "gt_dir": os.path.join(data_dir, "train", 'gt'),
                    "blur_dir": os.path.join(data_dir, "train", 'blur')
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "gt_dir": os.path.join(data_dir, "val", 'gt'),
                    "blur_dir": os.path.join(data_dir, "val", 'blur')
                },
            ),
        ]

    def _generate_examples(self, gt_dir, blur_dir):
        gt_files = sorted(os.listdir(gt_dir))
        blur_files = sorted(os.listdir(blur_dir))

        for idx in range(len(gt_files)):
            input_image_path = os.path.join(gt_dir, gt_files[idx])
            label_image_path = os.path.join(blur_dir, blur_files[idx])
            yield idx, {
                "gt": input_image_path,
                "blur": label_image_path,
            }