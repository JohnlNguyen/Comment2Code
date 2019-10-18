import os
import json
import pdb

pairs_dir = "../data/Pairs"
files = os.listdir(pairs_dir)

total = 0
data = []
for file in files:
    pdb.set_trace()

    with open(os.path.join(pairs_dir, file), 'r') as f:
        chunk = json.load(f)
        data.extend(chunk)

file_name = "../data/comment_code.json"

with open(file_name, 'w') as f:
    json.dump(data, f)

print('Total merged {}'.format(len(data)))
