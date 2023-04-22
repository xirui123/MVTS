import random

with open('LJ.csv', 'r') as f:
  lines = f.readlines()

num_lines = len(lines)
num_sample = int(num_lines * 0.05)
num_rest = num_lines - num_sample

random_indices = random.sample(range(num_lines), num_sample)

with open('LJ.Val.csv', 'w') as f:
  for i in random_indices:
    f.write(lines[i])

with open('LJ.Train.csv', 'w') as f:
  for i in range(num_lines):
    if i not in random_indices:
      f.write(lines[i])
