import os
from pathlib import Path

import PIL
import pandas as pd
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf, plot_spectrogram_and_pitch_and_energy_to_buf
from src.synthesis.synthesis import Synthesizer
from src.utils import inf_loop, MetricTracker
from torch.cuda.amp import GradScaler
from src.utils import optional_autocast
from src.waveglow import utils


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        log_step=400,  # how often WANDB will log
        log_predictions_step_epoch=5,
        mixed_precision=False,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model, criterion, metrics, optimizer, config, device, lr_scheduler
        )
        self.skip_oom = skip_oom
        self.train_dataloader = dataloaders["train"]
        self.config = config
        self.samplerate = 22050
        self.accumulation_steps = config["trainer"].get("accumulation_steps", 1)
        if len_epoch is None:
            self.len_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.log_predictions_step_epoch = log_predictions_step_epoch
        self.mixed_precision = mixed_precision
        self.train_metrics = MetricTracker(
            "loss",
            "mel_loss",
            "dp_loss",
            "pitch_loss",
            "energy_loss",
            "grad norm",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.waveglow = utils.get_WaveGlow().cuda()
        self.synthesizer = Synthesizer(
            waveglow=self.waveglow, device=self.device, samplerate=self.samplerate
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        tensor_types = {
            "src_seq": torch.int,
            "mel_target": torch.float,
            "duration": torch.long,
            "mel_pos": torch.int,
            "src_pos": torch.int,
            "pitch_target": torch.float,
            "energy_target": torch.float,
        }
        for tensor_name, tensor_type in tensor_types.items():
            if tensor_name in batch:
                batch[tensor_name] = batch[tensor_name].to(device).type(tensor_type)

        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()
        batch_idx = 0
        for i, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx=batch_idx,  #
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            for loss_type in [
                "loss",
                "mel_loss",
                "dp_loss",
                "energy_loss",
                "pitch_loss",
            ]:
                self.train_metrics.update(
                    loss_type, batch.get(loss_type, 0).detach().cpu().item()
                )
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx == 0:
                last_train_metrics = self.debug(batch, batch_idx, epoch)
            elif batch_idx >= self.len_epoch:
                break
            batch_idx += 1
            self.lr_scheduler.step()
        log = last_train_metrics
        if epoch % self.log_predictions_step_epoch == 0:
            print("Logging predictions!")
            self._log_predictions()
        return log

    @torch.no_grad()
    def debug(self, batch, batch_idx, epoch):
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.writer.add_scalar(
            "epoch",
            epoch,
        )
        self.logger.debug(
            "Train Epoch: {} {} Loss: {:.6f}".format(
                epoch,
                self._progress(batch_idx),
                self.train_metrics.avg("loss"),
            )
        )
        self.writer.add_scalar(
            "learning rate",
            self.optimizer.state_dict()["param_groups"][0]["lr"],
        )
        raw_text = " ".join(
            [
                self.synthesizer.inv_symbols.get(x, "<>")
                for x in batch["raw_text"][0].cpu().tolist()
            ]
        )
        audio, audio_waveglow = self.synthesizer.train_log(
            batch["mel"][0], batch["duration"][0]
        )
        self.writer.add_audio(
            "pred", audio, sample_rate=self.samplerate, caption=raw_text
        )
        self.writer.add_audio(
            "waveglow pred",
            audio_waveglow,
            sample_rate=self.samplerate,
            caption=raw_text,
        )
        audio_target, audio_waveglow = self.synthesizer.train_log(
            batch["mel_target"][0], batch["duration"][0]
        )
        self.writer.add_audio(
            "target", audio_target, sample_rate=self.samplerate, caption=raw_text
        )
        self.writer.add_audio(
            "waveglow target",
            audio_waveglow,
            sample_rate=self.samplerate,
            caption=raw_text,
        )
        try:
            self._log_spectrogram(batch)
        except Exception as e:
            print(f"Error displaying spectrogram: {e}. Continue.")
        self._log_scalars(self.train_metrics)
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()
        return last_train_metrics

    def process_batch(self, batch, batch_idx: int, metrics: MetricTracker):
        if batch_idx % self.accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
        batch = self.move_batch_to_device(batch, self.device)
        with optional_autocast(self.mixed_precision):
            outputs = self.model(**batch)
            batch.update(outputs)
            criterion = self.criterion(**batch)
            batch.update(criterion)
        self.scaler.scale(batch["loss"] / self.accumulation_steps).backward()
        if batch_idx % self.accumulation_steps == 0:
            self._clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def _log_predictions(self):
        self.model.eval()
        rows = {}
        for audio_path, audio_path_waveglow, params, text in zip(
            *self.synthesizer.create_audios(self.model)
        ):
            rows[Path(audio_path).name] = {
                "audio_waveglow": wandb.Audio(
                    audio_path_waveglow, sample_rate=self.samplerate
                ),
                "audio": wandb.Audio(audio_path, sample_rate=self.samplerate),
                "speed/pitch/energy": "{}/{}/{}".format(*params),
                "text": text,
            }
            if params == (1, 1, 1):
                self.writer.add_audio(
                    text[:5], audio_path_waveglow, sample_rate=self.samplerate
                )
        self.writer.add_table("synthesis", pd.DataFrame.from_dict(rows, orient="index"))

    @staticmethod
    def make_image(buff):
        return ToTensor()(PIL.Image.open(buff))
    #
    # @torch.no_grad()
    # def _log_spectrogram(self, batch):
    #     spectrogram_types = ["", "_target"]
    #     for spectrogram_type in spectrogram_types:
    #         spectrogram = (
    #             batch[f"mel{spectrogram_type}"][0]
    #             .detach()
    #             .cpu()
    #             .to(torch.float64)
    #             .transpose(1, 0)
    #         )
    #         spectrogram = torch.nan_to_num(spectrogram)
    #         self.writer.add_image(
    #             f"{spectrogram_type or 'pred'}",
    #             Trainer.make_image(plot_spectrogram_to_buf(spectrogram)),
    #         )
    @torch.no_grad()
    def _log_spectrogram(self, batch):
        spectrogram_types = ["", "_target"]
        for spectrogram_type in spectrogram_types:
            spectrogram = (
                batch[f"mel{spectrogram_type}"][0]
                .detach()
                .cpu()
                .to(torch.float64)
                .transpose(1, 0)
            )
            spectrogram = torch.nan_to_num(spectrogram)
            pitch = batch[f"pitch{spectrogram_type if spectrogram_type == '_target' else '_predicted'}"][0].detach().cpu()
            energy = batch[f"energy{spectrogram_type if spectrogram_type == '_target' else '_predicted'}"][0].detach().cpu()
            if spectrogram_type == '_target':
                pitch = pitch.log1p()
                energy = energy.log1p()
            pitch = pitch.numpy()
            energy = energy.numpy()
            buf = plot_spectrogram_and_pitch_and_energy_to_buf(spectrogram, pitch, energy)
            self.writer.add_image(
                f"{spectrogram_type or 'pred'}",
                Trainer.make_image(buf),
            )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(
                        # nan occurs in first batch in first run with grad scaler
                        torch.nan_to_num(p.grad, nan=0).detach(),
                        norm_type,
                    ).cpu()
                    for p in parameters
                ]
            ),
            norm_type,
        )
        return total_norm.item()

    @torch.no_grad()
    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(metric_name, metric_tracker.avg(metric_name))
