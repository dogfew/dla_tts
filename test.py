import argparse
import json
import os
from pathlib import Path

import torch
import src.model as module_model
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser
from src.waveglow import utils
from synthesis.synthesis import Synthesizer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main(config, args):
    logger = config.get_logger("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    waveglow = utils.get_WaveGlow().cuda()
    synthesizer = Synthesizer(waveglow=waveglow, device=device, dir='final_results')
    with torch.no_grad():
        if args.text is not None:
            raw_text = args.text
            preprocessed_phonemes = synthesizer.g2p(raw_text)
            preprocessed_phonemes = [item.replace('.', ' ')
                                     for item in preprocessed_phonemes if item != ' ']

            synthesizer(model, preprocessed_phonemes, idx=0)
        else:
            synthesizer.create_audios(model)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-o",
        "--out_dir",
        default=None,
        type=str,
        help="Where to save result outputs",
    )
    args.add_argument(
        '-t'
        '--text',
        default=None,
        type=str,
        help="Text to speech."
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirDataset",
                        "args": {
                            "dir": str(test_data_folder),
                        },
                    }
                ],
            }
        }
        print(config.config['data'])
    if config.config.get("data", {}).get("test", None) is not None:
        arg = 'test'
    elif config.config.get("data", {}).get("test-clean", None) is not None:
        arg = 'test-clean'
    elif config.config.get("data", {}).get("val", None) is not None:
        arg = 'val'
    else:
        raise AssertionError("Should provide test!")
    config["data"][arg]["batch_size"] = args.batch_size
    config["data"][arg]["n_jobs"] = args.jobs
    main(config, args)
