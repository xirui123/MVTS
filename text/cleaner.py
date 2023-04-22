import re

from text import cleaned_text_to_sequence
from text.en_frontend import en_to_phonemes
from text.symbols import symbols

source = ["-", "--"]
target = ["sp", "sp"]
mapping = dict(zip(source, target))


def remove_invalid_phonemes(phonemes):
  # 移除未识别的音素符号
  new_phones = []
  for ph in phonemes:
    ph = mapping.get(ph, ph)
    if ph in symbols:
      new_phones.append(ph)
    else:
      print("skip：", ph)
  return new_phones


def text_to_sequence(text):
  phones = text_to_phones(text)
  return cleaned_text_to_sequence(phones)


def text_to_phones(text: str) -> list:
  phonemes = en_to_phonemes(text)
  phonemes = remove_invalid_phonemes(phonemes)
  return phonemes
