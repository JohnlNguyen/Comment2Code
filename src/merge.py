import os
import pickle

from code_crawler import CommentCodeRow
pairs_dir = "../Data/Pairs"

files = os.listdir(pairs_dir)

total = 0
data = []
for file in files:
    with open(os.path.join(pairs_dir, file), 'rb') as f:
        data.extend(pickle.load(f))


with open('../Data/comment_code.pkl', 'wb') as f:
    pickle.dump(data, f)

print('Total merged {}'.format(len(data)))
