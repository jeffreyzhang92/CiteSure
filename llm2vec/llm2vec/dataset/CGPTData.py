import json
import random
import os

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

CGPT_EMBEDDING_PROMPTS = {
    #"final_contrastive_1": "Given a sentence and a biomedical paper represented by a title and abstract, predict whether the sentence cites the article.",
    #"final_contrastive_2": "Given a sentence and a biomedical paper represented by a title and abstract, predict whether the sentence cites the article.",
    #"final_contrastive_3": "Given a sentence and a biomedical paper represented by a title and abstract, predict whether the sentence cites the article.",
    #"final_contrastive_4": "Given a sentence and a biomedical paper represented by a title and abstract, predict whether the sentence cites the article.",
    #"final_contrastive_5": "Given a sentence and a biomedical paper represented by a title and abstract, predict whether the sentence cites the article.",
    "final_contrastive_test": "Given a sentence and a biomedical paper represented by a title and abstract, predict whether the sentence cites the article.",
}


class CGPTData(Dataset):
    def __init__(
        self,
        dataset_name: str = "CGPT",
        split: str = "validation",
        file_path: str = "cache/citationgpt",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading citationgpt data from {file_path}...")
        # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        for dataset in CGPT_EMBEDDING_PROMPTS:
            logger.info(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []
            with open(os.path.join(file_path, f"{dataset}.jsonl"), "r") as f:
                dataset_samples = f.readlines()

            dataset_samples = [json.loads(d) for d in dataset_samples]

            for i, sample in enumerate(dataset_samples):
                instruction = (
                    CGPT_EMBEDDING_PROMPTS[dataset]
                    if isinstance(CGPT_EMBEDDING_PROMPTS[dataset], str)
                    else CGPT_EMBEDDING_PROMPTS[dataset][i % 2]
                )
                query = f"{instruction}; " + self.separator + sample["query"]
                pos = self.separator + sample["positive"]
                neg = self.separator + sample["negative"]

                data_map[dataset].append(id_)

                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                )
                id_ += 1

        # combine split1 and split2
        new_data_map = {}
        for dataset in data_map:
            new_dataset = dataset.replace("_split1", "").replace("_split2", "")
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += data_map[dataset]
        data_map = new_data_map

        if self.shuffle_individual_datasets:
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching citationgpt data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "CGPTData does not have a validation split."
