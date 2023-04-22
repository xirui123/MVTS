# modified from https://github.com/jaywalnut310/vits
import argparse
import math
import os

import torch
from phaseaug.phaseaug import PhaseAug
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import commons
import utils
from data_utils import (TextAudioLoader, TextAudioCollate,
                        create_spec)
from losses import (generator_loss, discriminator_loss, feature_loss, kl_loss)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import (
  SynthesizerTrn,
  AvocodoDiscriminator)
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main(args):
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  hps = utils.get_hparams(args)
  # create spectrogram files
  create_spec(hps.data.training_files, hps.data)
  create_spec(hps.data.validation_files, hps.data)
  run(n_gpus, hps, args)


def count_parameters(model, scale=1000000):
  return sum(p.numel()
             for p in model.parameters() if p.requires_grad) / scale


def run(n_gpus, hps, args):
  global global_step

  logger = utils.get_logger(hps.model_dir)
  logger.info('MODEL NAME: {} in {}'.format(args.model, hps.model_dir))
  logger.info(
    'GPU: Use {} gpu(s) with batch size {} (FP16 running: {})'.format(
      n_gpus, hps.train.batch_size, hps.train.fp16_run))

  utils.check_git_hash(hps.model_dir)
  writer = SummaryWriter(log_dir=hps.model_dir)

  torch.manual_seed(hps.train.seed)

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data,
                                  args.initial_run)
  collate_fn = TextAudioCollate()

  eval_dataset = TextAudioLoader(hps.data.validation_files,
                                 hps.data, args.initial_run)
  eval_loader = DataLoader(
    eval_dataset,
    num_workers=2,
    shuffle=False,
    batch_size=hps.train.batch_size,
    drop_last=False,
    collate_fn=collate_fn,
  )

  logger.info('Training Started')

  train_loader = DataLoader(
    train_dataset,
    num_workers=8,
    shuffle=True,
    batch_size=hps.train.batch_size,
    collate_fn=collate_fn,
    pin_memory=True,
  )

  net_g = SynthesizerTrn(len(symbols),
                         hps.data.filter_length // 2 + 1,
                         hps.train.segment_size // hps.data.hop_length,
                         midi_start=hps.data.midi_start,
                         midi_end=hps.data.midi_end,
                         octave_range=hps.data.octave_range,
                         **hps.model).cuda()
  net_d = AvocodoDiscriminator(hps.model.use_spectral_norm).cuda()

  logger.info('MODEL SIZE: G {:.2f}M and D {:.2f}M'.format(
    count_parameters(net_g),
    count_parameters(net_d),
  ))

  optim_g = torch.optim.AdamW(net_g.parameters(),
                              hps.train.learning_rate,
                              betas=hps.train.betas,
                              eps=hps.train.eps)
  optim_d = torch.optim.AdamW(net_d.parameters(),
                              hps.train.learning_rate,
                              betas=hps.train.betas,
                              eps=hps.train.eps)

  if args.transfer:
    _, _, _, _, _, _, _ = utils.load_checkpoint(args.transfer, 0, net_g,
                                                net_d, None, None)
    epoch_str = 1
    global_step = 0

  elif args.force_resume:
    _, _, _, epoch_save, _ = utils.load_checkpoint_diffsize(
      args.force_resume, 0, net_g, net_d)
    epoch_str = 1
    global_step = 0
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
      optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
      optim_d, gamma=hps.train.lr_decay, last_epoch=-1)
  elif args.resume:
    _, _, _, _, _, epoch_save, _ = utils.load_checkpoint(
      args.resume, 0, net_g, net_d, optim_g, optim_d)
    epoch_str = epoch_save + 1
    global_step = epoch_save * len(train_loader) + 1
  else:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
    optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  outer_bar = tqdm(total=hps.train.epochs,
                   desc="Training",
                   position=0,
                   leave=False)
  outer_bar.update(epoch_str)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    train_and_evaluate(epoch, hps, [net_g, net_d],
                       [optim_g, optim_d], [scheduler_g, scheduler_d],
                       scaler, [train_loader, eval_loader], writer)

    scheduler_g.step()
    scheduler_d.step()
    outer_bar.update(1)


def train_and_evaluate(epoch, hps, nets, optims, schedulers, scaler,
                       loaders, writer):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  aug = PhaseAug().cuda()
  global global_step

  net_g.train()
  net_d.train()

  inner_bar = tqdm(total=len(train_loader),
                   desc="Epoch {}".format(epoch),
                   position=1,
                   leave=False)

  for batch_idx, (phonemes, phonemes_lengths,
                  spec, spec_lengths,
                  ying, ying_lengths,
                  wav, wav_lengths,
                  phndur) in enumerate(train_loader):

    phonemes, phonemes_lengths = phonemes.cuda(non_blocking=True), phonemes_lengths.cuda(non_blocking=True)
    spec, spec_lengths = spec.cuda(non_blocking=True), spec_lengths.cuda(non_blocking=True)
    ying, ying_lengths = ying.cuda(non_blocking=True), ying_lengths.cuda(non_blocking=True)

    wav, wav_lengths = wav.cuda(non_blocking=True), wav_lengths.cuda(non_blocking=True)

    phndur = phndur.cuda(non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, ids_slice, x_mask, z_mask, y_hat_, \
        (z, z_p, m_p, logs_p, m_q, logs_q), _, \
        (z_spec, m_spec, logs_spec, spec_mask, z_yin, m_yin, logs_yin, yin_mask), \
        (yin_gt_crop, yin_gt_shifted_crop, yin_dec_crop, yin_hat_crop, scope_shift, yin_hat_shifted) \
        = net_g(
        phonemes, phonemes_lengths, spec, spec_lengths, ying, ying_lengths, phndur
      )

      mel = spec_to_mel_torch(spec, hps.data.filter_length,
                              hps.data.n_mel_channels,
                              hps.data.sampling_rate, hps.data.mel_fmin,
                              hps.data.mel_fmax)
      y_mel = commons.slice_segments(
        mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
        y_hat[-1].squeeze(1), hps.data.filter_length,
        hps.data.n_mel_channels, hps.data.sampling_rate,
        hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin,
        hps.data.mel_fmax)
      yin_gt_crop = commons.slice_segments(
        torch.cat([yin_gt_crop, yin_gt_shifted_crop], dim=0),
        ids_slice, hps.train.segment_size // hps.data.hop_length)

      y_ = commons.slice_segments(torch.cat([wav, wav], dim=0),
                                  ids_slice * hps.data.hop_length,
                                  hps.train.segment_size)  # sliced
      # Discriminator
      with autocast(enabled=False):
        aug_y_, aug_y_hat_last = aug.forward_sync(
          y_, y_hat_[-1].detach())
        aug_y_hat_ = [_y.detach() for _y in y_hat_[:-1]]
        aug_y_hat_.append(aug_y_hat_last)
      y_d_hat_r, y_d_hat_g, _, _ = net_d(aug_y_, aug_y_hat_)
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
          y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    p = float(batch_idx + epoch *
              len(train_loader)) / hps.train.alpha / len(train_loader)
    alpha = 2. / (1. + math.exp(-20 * p)) - 1

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      with autocast(enabled=False):
        aug_y_, aug_y_hat_last = aug.forward_sync(y_, y_hat_[-1])
        aug_y_hat_ = y_hat_
        aug_y_hat_[-1] = aug_y_hat_last
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(aug_y_, aug_y_hat_)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                          z_mask) * hps.train.c_kl
        loss_yin_dec = F.l1_loss(yin_gt_shifted_crop,
                                 yin_dec_crop) * hps.train.c_yin
        loss_yin_shift = F.l1_loss(
          torch.exp(-yin_gt_crop),
          torch.exp(-yin_hat_crop)) * hps.train.c_yin + F.l1_loss(
          torch.exp(-yin_hat_shifted),
          torch.exp(-(torch.chunk(yin_hat_crop, 2, dim=0)[1]))
        ) * hps.train.c_yin
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_yin_shift + loss_yin_dec
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    inner_bar.update(1)
    inner_bar.set_description(
      "Epoch {} | g {: .04f} d {: .04f}| {: d} step".format(
        epoch, loss_gen_all, loss_disc_all, global_step))
    if global_step % hps.train.log_interval == 0:
      lr = optim_g.param_groups[0]['lr']

      scalar_dict = {
        "learning_rate": lr,
        "loss/g/score": sum(losses_gen),
        "loss/g/fm": loss_fm,
        "loss/g/mel": loss_mel,
        "loss/g/dur": loss_dur,
        "loss/g/kl": loss_kl,
        "loss/g/yindec": loss_yin_dec,
        "loss/g/yinshift": loss_yin_shift,
        "loss/g/total": loss_gen_all,
        "loss/d/real": sum(losses_disc_r),
        "loss/d/gen": sum(losses_disc_g),
        "loss/d/total": loss_disc_all,
      }

      utils.summarize(writer=writer,
                      global_step=global_step,
                      scalars=scalar_dict)

    if global_step % hps.train.eval_interval == 0:
      evaluate(hps, global_step, epoch, net_g, eval_loader, writer)

    global_step += 1

  if epoch % hps.train.save_interval == 0:
    utils.save_checkpoint(
      net_g, optim_g, net_d, optim_d, hps, epoch,
      hps.train.learning_rate,
      os.path.join(hps.model_dir,
                   "{}_{}.pth".format(hps.model_name, epoch)))


def evaluate(hps, current_step, epoch, generator, eval_loader, writer):
  generator.eval()
  n_sample = hps.train.n_sample
  with torch.no_grad():
    loss_val_mel = 0
    loss_val_yin = 0
    val_bar = tqdm(total=len(eval_loader),
                   desc="Validation (Step {})".format(current_step),
                   position=1,
                   leave=False)

    for batch_idx, (phonemes, phonemes_lengths,
                    spec, spec_lengths,
                    ying, ying_lengths,
                    wav, wav_lengths,
                    phndur) in enumerate(eval_loader):

      phonemes, phonemes_lengths = phonemes.cuda(0, non_blocking=True), phonemes_lengths.cuda(
        0, non_blocking=True)
      spec, spec_lengths = spec.cuda(
        0, non_blocking=True), spec_lengths.cuda(0, non_blocking=True)
      ying, ying_lengths = ying.cuda(
        0, non_blocking=True), ying_lengths.cuda(0, non_blocking=True)
      wav, wav_lengths = wav.cuda(0, non_blocking=True), wav_lengths.cuda(
        0, non_blocking=True)
      phndur = phndur.cuda(0, non_blocking=True)

      with autocast(enabled=hps.train.fp16_run):
        y_hat, l_length, ids_slice, x_mask, z_mask, y_hat_, \
          (z, z_p, m_p, logs_p, m_q, logs_q), \
          _, \
          (z_spec, m_spec, logs_spec, spec_mask, z_yin, m_yin, logs_yin, yin_mask), \
          (yin_gt_crop, yin_gt_shifted_crop, yin_dec_crop, yin_hat_crop, scope_shift, yin_hat_shifted) \
          = generator(
          phonemes, phonemes_lengths, spec, spec_lengths, ying, ying_lengths, phndur
        )

        mel = spec_to_mel_torch(spec, hps.data.filter_length,
                                hps.data.n_mel_channels,
                                hps.data.sampling_rate,
                                hps.data.mel_fmin, hps.data.mel_fmax)
        y_mel = commons.slice_segments(
          mel, ids_slice,
          hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
          y_hat[-1].squeeze(1), hps.data.filter_length,
          hps.data.n_mel_channels, hps.data.sampling_rate,
          hps.data.hop_length, hps.data.win_length,
          hps.data.mel_fmin, hps.data.mel_fmax)
        with autocast(enabled=False):
          loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
          loss_val_mel += loss_mel.item()
          loss_yin = F.l1_loss(yin_gt_shifted_crop,
                               yin_dec_crop) * hps.train.c_yin
          loss_val_yin += loss_yin.item()

      if batch_idx == 0:
        phonemes = phonemes[:n_sample]
        phonemes_lengths = phonemes_lengths[:n_sample]
        spec = spec[:n_sample]
        spec_lengths = spec_lengths[:n_sample]
        ying = ying[:n_sample]
        ying_lengths = ying_lengths[:n_sample]
        wav = wav[:n_sample]
        wav_lengths = wav_lengths[:n_sample]

        decoder_inputs, mask, (z_crop, z, *_) = generator.infer_pre_decoder(phonemes, phonemes_lengths, max_len=2000)
        y_hat = generator.infer_decode_chunk(decoder_inputs)

        # scope-shifted
        z_spec, z_yin = torch.split(z,
                                    hps.model.inter_channels -
                                    hps.model.yin_channels,
                                    dim=1)
        z_yin_crop = generator.crop_scope([z_yin], 6)[0]
        z_crop_shift = torch.cat([z_spec, z_yin_crop], dim=1)
        decoder_inputs_shift = z_crop_shift * mask
        y_hat_shift = generator.infer_decode_chunk(decoder_inputs_shift)
        z_yin = z_yin * mask
        yin_hat = generator.yin_dec_infer(z_yin, mask)

        y_hat_mel_length = mask.sum([1, 2]).long()
        y_hat_lengths = y_hat_mel_length * hps.data.hop_length

        mel = spec_to_mel_torch(spec, hps.data.filter_length,
                                hps.data.n_mel_channels,
                                hps.data.sampling_rate,
                                hps.data.mel_fmin, hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1).float(), hps.data.filter_length,
          hps.data.n_mel_channels, hps.data.sampling_rate,
          hps.data.hop_length, hps.data.win_length,
          hps.data.mel_fmin, hps.data.mel_fmax)
        y_hat_shift_mel = mel_spectrogram_torch(
          y_hat_shift.squeeze(1).float(), hps.data.filter_length,
          hps.data.n_mel_channels, hps.data.sampling_rate,
          hps.data.hop_length, hps.data.win_length,
          hps.data.mel_fmin, hps.data.mel_fmax)
        y_hat_pad = F.pad(
          y_hat, (hps.data.filter_length - hps.data.hop_length,
                  hps.data.filter_length - hps.data.hop_length +
                  (-y_hat.shape[-1]) % hps.data.hop_length +
                  hps.data.hop_length *
                  (y_hat.shape[-1] % hps.data.hop_length == 0)),
          mode='reflect').squeeze(1)
        y_hat_shift_pad = F.pad(
          y_hat_shift,
          (hps.data.filter_length - hps.data.hop_length,
           hps.data.filter_length - hps.data.hop_length +
           (-y_hat.shape[-1]) % hps.data.hop_length +
           hps.data.hop_length *
           (y_hat.shape[-1] % hps.data.hop_length == 0)),
          mode='reflect').squeeze(1)
        ying_hat = generator.pitch.yingram(y_hat_pad)
        ying_hat_shift = generator.pitch.yingram(
          y_hat_shift_pad)

        if y_hat_mel.size(2) < mel.size(2):
          zero = torch.full((n_sample, y_hat_mel.size(1),
                             mel.size(2) - y_hat_mel.size(2)),
                            -11.5129).to(y_hat_mel.device)
          y_hat_mel = torch.cat((y_hat_mel, zero), dim=2)
          y_hat_shift_mel = torch.cat((y_hat_shift_mel, zero), dim=2)
          zero = torch.full((n_sample, yin_hat.size(1),
                             mel.size(2) - yin_hat.size(2)),
                            0).to(y_hat_mel.device)
          yin_hat = torch.cat((yin_hat, zero), dim=2)
          zero = torch.full((n_sample, ying_hat.size(1),
                             mel.size(2) - ying_hat.size(2)),
                            0).to(y_hat_mel.device)
          ying_hat = torch.cat((ying_hat, zero), dim=2)
          ying_hat_shift = torch.cat((ying_hat_shift, zero), dim=2)
          zero = torch.full(
            (n_sample, z_yin.size(1), mel.size(2) - z_yin.size(2)),
            0).to(y_hat_mel.device)
          z_yin = torch.cat((z_yin, zero), dim=2)

          ids = torch.arange(0, mel.size(2)).unsqueeze(0).expand(
            mel.size(1),
            -1).unsqueeze(0).expand(n_sample, -1,
                                    -1).to(y_hat_mel_length.device)
          mask = ids > y_hat_mel_length.unsqueeze(1).expand(
            -1, mel.size(1)).unsqueeze(2).expand(
            -1, -1, mel.size(2))
          y_hat_mel.masked_fill_(mask, -11.5129)
          y_hat_shift_mel.masked_fill_(mask, -11.5129)

        image_dict = dict()
        audio_dict = dict()
        for i in range(n_sample):
          image_dict.update({
            "gen/{}_mel".format(i):
              utils.plot_spectrogram_to_numpy(
                y_hat_mel[i].cpu().numpy())
          })
          audio_dict.update({
            "gen/{}_audio".format(i):
              y_hat[i, :, :y_hat_lengths[i]]
          })
          image_dict.update({
            "gen/{}_mel_shift".format(i):
              utils.plot_spectrogram_to_numpy(
                y_hat_shift_mel[i].cpu().numpy())
          })
          audio_dict.update({
            "gen/{}_audio_shift".format(i):
              y_hat_shift[i, :, :y_hat_lengths[i]]
          })
          image_dict.update({
            "gen/{}_z_yin".format(i):
              utils.plot_spectrogram_to_numpy(z_yin[i].cpu().numpy())
          })
          image_dict.update({
            "gen/{}_yin_dec".format(i):
              utils.plot_spectrogram_to_numpy(
                yin_hat[i].cpu().numpy())
          })
          image_dict.update({
            "gen/{}_ying".format(i):
              utils.plot_spectrogram_to_numpy(
                ying_hat[i].cpu().numpy())
          })
          image_dict.update({
            "gen/{}_ying_shift".format(i):
              utils.plot_spectrogram_to_numpy(
                ying_hat_shift[i].cpu().numpy())
          })

        if current_step == 0:
          for i in range(n_sample):
            image_dict.update({
              "gt/{}_mel".format(i):
                utils.plot_spectrogram_to_numpy(
                  mel[i].cpu().numpy())
            })
            image_dict.update({
              "gt/{}_ying".format(i):
                utils.plot_spectrogram_to_numpy(
                  ying[i].cpu().numpy())
            })
            audio_dict.update(
              {"gt/{}_audio".format(i): wav[i, :, :wav_lengths[i]]})

        utils.summarize(writer=writer,
                        global_step=epoch,
                        images=image_dict,
                        audios=audio_dict,
                        audio_sampling_rate=hps.data.sampling_rate)
      val_bar.update(1)
    loss_val_mel = loss_val_mel / len(eval_loader)
    loss_val_yin = loss_val_yin / len(eval_loader)

    scalar_dict = {
      "loss/val/mel": loss_val_mel,
      "loss/val/yin": loss_val_yin,
    }
    utils.summarize(writer=writer,
                    global_step=current_step,
                    scalars=scalar_dict)
  generator.train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-c',
                      '--config',
                      type=str,
                      default="./configs/config_en.yaml",
                      help='Path to configuration file')
  parser.add_argument('-m',
                      '--model',
                      type=str,
                      default="LJ",
                      help='Model name')
  parser.add_argument('-r',
                      '--resume',
                      type=str,
                      help='Path to checkpoint for resume')
  parser.add_argument('-f',
                      '--force_resume',
                      type=str,
                      help='Path to checkpoint for force resume')
  parser.add_argument('-t',
                      '--transfer',
                      type=str,
                      help='Path to baseline checkpoint for transfer')
  parser.add_argument('-w',
                      '--ignore_warning',
                      action="store_true",
                      help='Ignore warning message')
  parser.add_argument('-i',
                      '--initial_run',
                      action="store_true",
                      help='Inintial run for saving pt files')
  args = parser.parse_args()
  if args.ignore_warning:
    import warnings

    warnings.filterwarnings(action='ignore')

  main(args)
