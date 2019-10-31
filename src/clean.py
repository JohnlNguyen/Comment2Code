from pdb import set_trace
import json
import itertools
from nltk.tokenize import RegexpTokenizer

data = []
with open('../data/Pairs/code_comment_93843.json', 'r') as f:
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


def split_string(s):
	return list(itertools.chain(*[w.strip().split(' ') for w in s.split('\\n') if w]))


def is_different(a, b, n):
	b = split_string(b)
	a = split_string(a)

	if abs(len(a) - len(b)) >= n:
		return True

	return sum([1 if x != y else 0 for x, y in zip(a, b)]) >= n


def tokenize_code(line):
	return RegexpTokenizer(r'\w+').tokenize(line)


def comment_contains_code(comment, code):
	comment = split_string(comment)
	code = tokenize_code("\\n".join(code))
	code = split_string(" ".join(code))
	intersection = set(comment).intersection(set(code))
	return len(intersection)


def clean_code(code):
	code = [c.strip() for c in code]
	return code


def clean_comment(comment):
	comment = comment.split("\\n")
	comment = [c.replace("#", "").strip() for c in comment if c]
	return "\\n".join(comment)


for row in data:
	if row['before_heuristic'] == "BLANK_LINE":
		row['before_code'] = remove_blank(row['before_code'])
	if row['after_heuristic'] == "BLANK_LINE":
		row['after_code'] = remove_blank(row['after_code'])

	if row['type'] == 'BOTH':
		if (row['before_comment'] == row['after_comment']
			or row['before_code'] == row['after_code']):
			row['type'] = 'COMMENT'

		# if not is_different(row['before_comment'], row['after_comment'], n=0):
		#     row['type'] = 'COMMENT'

		# if (comment_contains_code(row['before_comment'], row['before_code']) > 0 or
		#     comment_contains_code(row['after_comment'], row['after_code']) > 0):
		#     c += 1

		row['before_code'] = clean_code(row['before_code'])
		row['after_code'] = clean_code(row['after_code'])
		row['before_comment'] = clean_comment(row['before_comment'])
		row['after_comment'] = clean_comment(row['after_comment'])

	if not row['before_code'] or not row['after_code']:
		row['type'] = 'COMMENT'

print("-" * 100)
print("Both {}".format(sum([1 for row in data if row['type'] == "BOTH"])))
print("Comment {}".format(sum([1 for row in data if row['type'] == "COMMENT"])))
print("Total {}".format(len(data)))

with open('../data/Pairs/code_comment_clean.json', 'w') as f:
	json.dump([row for row in data if row['type'] == 'BOTH'], f)
