from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import src.datasets
from src import batch_sampler as batch_sampler_module
from src.collate_fn.collate import collate_fn
from src.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser, audio):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == "train":
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(
                configs.init_obj(
                    ds,
                    src.datasets,
                    audio=audio,
                    config_parser=configs,
                )
            )
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor(
            "batch_size" in params, "batch_sampler" in params
        ), "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            bs_expand = params["batch_expand_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(
                params["batch_sampler"], batch_sampler_module, data_source=dataset
            )
            bs, shuffle = 1, False
            bs_expand = 1
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(
            dataset
        ), f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        if batch_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=bs * bs_expand,
                collate_fn=lambda x: collate_fn(x, bs_expand),
                shuffle=shuffle,
                pin_memory=True,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                drop_last=True,
            )
        else:
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
            )
        dataloaders[split] = dataloader
    return dataloaders
