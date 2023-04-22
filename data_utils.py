# modified from https://github.com/jaywalnut310/vits
import os
import random

import torch
import torch.utils.data
from tqdm import tqdm

from analysis import Pitch
from mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text
import torch.nn.functional as F


class TextAudioLoader(torch.utils.data.Dataset):
  """
      1) loads audio, speaker_id, text pairs
      2) normalizes text and converts them to sequences of integers
      3) computes spectrograms from audio files.
  """

  def __init__(self, audiopaths_sid_text, hparams, pt_run=False):
    self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
    self.sampling_rate = hparams.sampling_rate
    self.filter_length = hparams.filter_length
    self.hop_length = hparams.hop_length
    self.win_length = hparams.win_length

    self.add_blank = hparams.add_blank
    self.min_text_len = getattr(hparams, "min_text_len", 1)
    self.max_text_len = getattr(hparams, "max_text_len", 190)

    self.data_path = hparams.data_path

    self.pitch = Pitch(sr=hparams.sampling_rate,
                       W=hparams.tau_max,
                       tau_max=hparams.tau_max,
                       midi_start=hparams.midi_start,
                       midi_end=hparams.midi_end,
                       octave_range=hparams.octave_range)

    random.seed(1234)
    random.shuffle(self.audiopaths_sid_text)
    self._filter()
    if pt_run:
      for _audiopaths_sid_text in self.audiopaths_sid_text:
        _ = self.get_audio_text_pair(_audiopaths_sid_text,
                                     True)

  def _filter(self):
    """
    Filter text & store spec lengths
    """
    # Store spectrogram lengths for Bucketing
    # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    # spec_length = wav_length // hop_length

    audiopaths_sid_text_new = []
    lengths = []
    for id_, phonemes, durations in self.audiopaths_sid_text:
      if self.min_text_len <= len(phonemes) <= self.max_text_len:
        wav_path = os.path.join(self.data_path, id_) + ".wav"
        audiopaths_sid_text_new.append([wav_path, phonemes, durations])
        lengths.append(os.path.getsize(wav_path) // (2 * self.hop_length))

    self.audiopaths_sid_text = audiopaths_sid_text_new
    self.lengths = lengths

  def get_audio_text_pair(self, audiopath_and_text, pt_run=False):
    wav_path, phonemes, durations = audiopath_and_text
    phonemes = self.get_phonemes(phonemes)
    phn_dur = self.get_duration_flag(durations)

    spec, ying, wav = self.get_audio(wav_path, pt_run)

    sumdur = sum(phn_dur)
    assert abs(spec.shape[-1] - sumdur) < 2, wav_path

    if spec.shape[-1] > sumdur:
      spec = spec[:, :sumdur]
      wav = wav[:, :sumdur * self.hop_length]
    elif spec.shape[-1] < sumdur:
      spec_pad = torch.zeros([spec.shape[0], sumdur])
      spec_pad[:, :spec.shape[-1]] = spec
      spec = spec_pad
      wav_pad = torch.zeros([1, sumdur * self.hop_length])
      wav_pad[:, :wav.shape[-1]] = wav
      wav = wav_pad

    if ying.shape[-1] > sumdur:
      ying = ying[:, :sumdur]
    elif ying.shape[-1] < sumdur:
      ying_pad = torch.zeros([ying.shape[0], sumdur])
      ying_pad[:, :ying.shape[-1]] = ying
      ying = ying_pad

    assert phonemes.shape == phn_dur.shape, (phonemes.shape, phn_dur.shape, wav_path)

    assert sumdur == wav.shape[-1] // self.hop_length == spec.shape[-1] == ying.shape[-1], \
           (sumdur, wav.shape[-1] // self.hop_length, spec.shape[-1], ying.shape[-1])

    return phonemes, spec, ying, wav, phn_dur

  def get_audio(self, filename, pt_run=False):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != self.sampling_rate:
      raise ValueError("{} SR doesn't match target {} SR".format(
        sampling_rate, self.sampling_rate))
    audio_norm = audio.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")
    ying_filename = filename.replace(".wav", ".ying.pt")
    if os.path.exists(spec_filename) and not pt_run:
      spec = torch.load(spec_filename, map_location='cpu')
    else:
      spec = spectrogram_torch(audio_norm,
                               self.filter_length,
                               self.sampling_rate,
                               self.hop_length,
                               self.win_length,
                               center=False)
      spec = torch.squeeze(spec, 0)
      torch.save(spec, spec_filename)
    if os.path.exists(ying_filename) and not pt_run:
      ying = torch.load(ying_filename, map_location='cpu')
    else:
      wav = torch.nn.functional.pad(
        audio_norm.unsqueeze(0),
        (self.filter_length - self.hop_length,
         self.filter_length - self.hop_length +
         (-audio_norm.shape[1]) % self.hop_length + self.hop_length * (audio_norm.shape[1] % self.hop_length == 0)),
        mode='constant').squeeze(0)
      ying = self.pitch.yingram(wav)[0]
      torch.save(ying, ying_filename)
    return spec, ying, audio_norm

  def get_phonemes(self, phonemes):
    text_norm = cleaned_text_to_sequence(phonemes.split(" "))
    text_norm = torch.LongTensor(text_norm)
    return text_norm

  def get_duration_flag(self, phn_dur):
    phn_dur = [int(i) for i in phn_dur.split(" ")]
    phn_dur = torch.LongTensor(phn_dur)
    return phn_dur

  def __getitem__(self, index):
    return self.get_audio_text_pair(
      self.audiopaths_sid_text[index])

  def __len__(self):
    return len(self.audiopaths_sid_text)


class TextAudioCollate:
  """ Zero-pads model inputs and targets"""

  def __init__(self, return_ids=False):
    self.return_ids = return_ids

  def __call__(self, batch):
    """Collate's training batch from normalized text, audio and speaker identities
    PARAMS
    ------
    batch: [text_normalized, spec_normalized, wav_normalized, sid]
    """
    # Right zero-pad all one-hot text sequences to max input length
    _, ids_sorted_decreasing = torch.sort(torch.LongTensor(
      [x[1].size(1) for x in batch]),
      dim=0,
      descending=True)

    # phonemes, spec, ying, wav, phn_dur
    max_phonemes_len = max([len(x[0]) for x in batch])
    max_spec_len = max([x[1].size(1) for x in batch])
    max_ying_len = max([x[2].size(1) for x in batch])
    max_wav_len = max([x[3].size(1) for x in batch])
    max_phndur_len = max([len(x[4]) for x in batch])

    phonemes_lengths = torch.LongTensor(len(batch))
    spec_lengths = torch.LongTensor(len(batch))
    ying_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))

    phonemes_padded = torch.LongTensor(len(batch), max_phonemes_len)
    spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0),
                                    max_spec_len)
    ying_padded = torch.FloatTensor(len(batch), batch[0][2].size(0),
                                    max_ying_len)
    wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
    phndur_padded = torch.LongTensor(len(batch), max_phndur_len)

    phonemes_padded.zero_()
    spec_padded.zero_()
    ying_padded.zero_()
    wav_padded.zero_()
    phndur_padded.zero_()

    for i in range(len(ids_sorted_decreasing)):
      row = batch[ids_sorted_decreasing[i]]

      phonemes = row[0]
      phonemes_padded[i, :phonemes.size(0)] = phonemes
      phonemes_lengths[i] = phonemes.size(0)

      spec = row[1]
      spec_padded[i, :, :spec.size(1)] = spec
      spec_lengths[i] = spec.size(1)

      ying = row[2]
      ying_padded[i, :, :ying.size(1)] = ying
      ying_lengths[i] = ying.size(1)

      wav = row[3]
      wav_padded[i, :, :wav.size(1)] = wav
      wav_lengths[i] = wav.size(1)

      phndur = row[4]
      phndur_padded[i, :phndur.size(0)] = phndur

    return phonemes_padded, phonemes_lengths, \
      spec_padded, spec_lengths, \
      ying_padded, ying_lengths, \
      wav_padded, wav_lengths, \
      phndur_padded


def create_spec(audiopaths_sid_text, hparams):
  audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
  for audiopath, _, _ in tqdm(audiopaths_sid_text):
    audiopath = os.path.join(hparams.data_path, audiopath) + ".wav"
    audio, sampling_rate = load_wav_to_torch(audiopath)
    if sampling_rate != hparams.sampling_rate:
      raise ValueError("{} SR doesn't match target {} SR".format(
        sampling_rate, hparams.sampling_rate))
    audio_norm = audio.unsqueeze(0)
    specpath = audiopath.replace(".wav", ".spec.pt")

    if not os.path.exists(specpath):
      spec = spectrogram_torch(audio_norm,
                               hparams.filter_length,
                               hparams.sampling_rate,
                               hparams.hop_length,
                               hparams.win_length,
                               center=False)
      spec = torch.squeeze(spec, 0)
      torch.save(spec, specpath)

def pad(input_ele):
  max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

  out_list = list()
  for i, batch in enumerate(input_ele):
    if len(batch.shape) == 1:
      one_batch_padded = F.pad(
        batch, (0, max_len - batch.size(0)), "constant", 0.0
      )
    elif len(batch.shape) == 2:
      one_batch_padded = F.pad(
        batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
      )
    out_list.append(one_batch_padded)
  out_padded = torch.stack(out_list)

  return out_padded
