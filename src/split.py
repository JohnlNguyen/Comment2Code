import json
import random

from pdb import set_trace


def swap_keys(label, line):
    if label == 0:
        if line["type"] == "BOTH":
            swap_dir = round(random.random())
            comment, code = ("before_comment", "after_code") if swap_dir == 0 else (
                "after_comment", "before_code")
        else:
            comment, code = "before_comment", "after_code"
    else:
        if line["type"] == "BOTH":
            swap_dir = round(random.random())
            comment, code = ("before_comment", "before_code") if swap_dir == 0 else (
                "after_comment", "after_code")
        else:
            comment, code = "after_comment", "after_code"
    return comment, code

data = []
with open('../data/Pairs/code_comment_clean.json', 'r') as f:
    data = json.load(f)


to_grade = []
for row in random.sample(data, 25):
    label = round(random.random())

    comment_k, code_k = swap_keys(label, row)
    swap = random.choice(data)
    code = swap[code_k] if label == 0 else row[code_k]
    comment = swap[comment_k] if label == 0 else row[comment_k]

    # comment, code = row[comment_k], row[code_k]

    org, proj, _, file = row['after_path'].split("#") if label != 0 else swap['after_path'].split("#")

    commit = row['commit'] if label != 0 else swap['commit']
    line = row['after_line'] if label != 0 else swap['after_line']


    link = "https://github.com/{}/{}/commit/{}".format(org, proj, commit)

    r = {'label': label, 'comment': comment, 'code': code, 'comment_key': comment_k, 'code_key': code_k, 'commit': commit,
         'link': link, 'file': file.replace("__", "/"), 'line': line}
    to_grade.append(r)


with open('../data/inspect_random.json', 'w') as f:
    json.dump(to_grade, f)
