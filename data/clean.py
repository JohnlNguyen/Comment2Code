from pdb import set_trace
import json

data = []
with open('Pairs/code_comment_90742.json', 'r') as f:
    data = json.load(f)

print("Both {}".format(sum([1 for row in data if row['type'] == "BOTH"])))
print("Comment {}".format(sum([1 for row in data if row['type'] == "COMMENT"])))
print("Total {}".format(len(data)))


def remove_blank(code):
    for i, line in enumerate(code):
        if line == "\n":
            code = code[:i]
            break
    return code

for row in data:
    if row['before_heuristic'] == "BLANK_LINE":
            row['before_code'] = remove_blank(row['before_code'])
    if row['after_heuristic'] == "BLANK_LINE":
        row['after_code'] = remove_blank(row['after_code'])

    row['before_comment'] = row['before_comment'].replace("#", "")
    row['after_comment'] = row['after_comment'].replace("#", "")

    if row['type'] == 'BOTH':
        if (row['before_comment'] == row['after_comment']
                or row['before_code'] == row['after_code']):
            row['type'] = 'COMMENT'

    if not row['before_code'] or not row['after_code']:
        row['type'] = 'COMMENT'

print("-"*100)
print("Both {}".format(sum([1 for row in data if row['type'] == "BOTH"])))
print("Comment {}".format(sum([1 for row in data if row['type'] == "COMMENT"])))
print("Total {}".format(len(data)))


with open('Pairs/code_comment_90742.json', 'w') as f:
    json.dump(data, f)
