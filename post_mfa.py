import os
import random

import numpy as np
import tgt
from tqdm import tqdm

silence_symbol = ["sil", "sp", "spn"]
sampling_rate = 22050
hop_length = 256


def sample(probabilities):
  probabilities = np.maximum(probabilities, 0)
  normalized_probs = probabilities / np.sum(probabilities)
  return np.random.choice(len(probabilities), p=normalized_probs)


def get_probability(x, minimum, maximum, mean):
  if x <= minimum or x >= maximum:
    return 0
  if x == mean:
    return 1
  if x < mean:
    return (x - minimum) / (mean - minimum)
  if x > mean:
    return (maximum - x) / (maximum - mean)


def get_sp(frames, is_last, is_first):
  if is_first:
    return "sp"
  if is_last:
    if random.random() < 0.8:
      return "sp"
    else:
      return "."
  pu_dict = {
    ",": [3, 15, 40],
    "…": [30, 1000, 1000]
  }
  probabilities = []
  for i in [",", "…"]:
    probabilities.append(get_probability(frames, *pu_dict[i]))
  probabilities.append(0.01)
  return [",", "…", "sp"][sample(probabilities)]


def get_alignment(tier):
  phones = []
  durations = []
  end_time = []
  last_end = 0
  for t in tier._objects:
    start, end, phone = t.start_time, t.end_time, t.text

    if last_end != start:
      durations.append(
        int(
          np.round(start * sampling_rate / hop_length)
          - np.round(last_end * sampling_rate / hop_length)
        )
      )
      phones.append('sp')
      end_time.append(start)

    phones.append(phone)
    durations.append(
      int(
        np.round(end * sampling_rate / hop_length)
        - np.round(start * sampling_rate / hop_length)
      )
    )
    end_time.append(end)

    last_end = end

  if tier.end_time != last_end:
    durations.append(
      int(
        np.round(tier.end_time * sampling_rate / hop_length)
        - np.round(last_end * sampling_rate / hop_length)
      )
    )
    phones.append('sp')
    end_time.append(tier.end_time)
  return phones, durations, end_time


def remove_dup(phs, dur):
  new_phos = []
  new_gtdurs = []
  last_ph = None
  for ph, dur in zip(phs, dur):
    if ph != last_ph:
      new_phos.append(ph)
      new_gtdurs.append(dur)
    else:
      new_gtdurs[-1] += dur
    last_ph = ph
  return new_phos, new_gtdurs


def refine(phones, durations):
  phones, durations = remove_dup(phones, durations)
  for idx in range(len(phones)):
    ph: str = phones[idx]
    dur = durations[idx]
    if ph.lower() in silence_symbol:
      phones[idx] = get_sp(dur, idx == len(phones) - 1 and phones[idx - 1] not in silence_symbol, idx == 0)

  return phones, durations


with open("filelists/LJ.csv", "w") as out_file:
  align_root = "mfa"
  for name in tqdm(sorted(os.listdir(align_root))):
    if name.endswith("Grid"):
      textgrid = tgt.io.read_textgrid(f"{align_root}/{name}")
      phone, duration, end_times = get_alignment(
        textgrid.get_tier_by_name("phones")
      )
      id_ = name.replace(".TextGrid", "")
      phone = ["sp" if ph in silence_symbol else ph for ph in phone]

      try:
        phone, duration = refine(phone, duration)
      except:
        print("错误，请检查：", align_root, name)
        continue

      ph = " ".join(phone)
      du = " ".join([str(i) for i in duration])
      out_file.write(f"{id_}|{ph}|{du}\n")
