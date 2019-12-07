from pdb import set_trace
import json
import itertools
# import pygments
import random
import csv
# from pygments.token import Comment, Text, Keyword, Name

# from bpe import Encoder
# from nltk import everygrams, ngrams
# from nltk.tokenize import word_tokenize


# from lexer import build_lexer


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


def tokenize_code(line, encoder):
    return set(encoder.tokenize(line)) - {encoder.EOW, encoder.SOW, encoder.UNK, encoder.PAD}


def comment_contains_code(comment, code, lexer, n=5):
    t_cod = []
    for line in code:
        t_cod.extend([x[1]
                      for x in list(pygments.lex(line, lexer)) if x[0] in Name])

    # t_cod = " ".join(code)
    t_cod = " ".join(t_cod)
    t_com = ["".join(x) for x in ngrams(comment.replace("\\n", " "), n)]
    t_cod = ["".join(x) for x in ngrams(t_cod, n)]
    intersection = set(t_com).intersection(set(t_cod))
    return len(intersection) > 0


def clean_code(code):
    code = [c.strip() for c in code]
    return code


def clean_comment(comment):
    comment = comment.split("\\n")
    comment = [c.replace("//", "").strip() for c in comment if c]
    return "\\n".join(comment)


def print_info(data):
    print("-" * 100)
    print("Both {}".format(sum([1 for row in data if row['type'] == "BOTH"])))
    print("Comment {}".format(
        sum([1 for row in data if row['type'] == "COMMENT"])))
    print("Code {}".format(sum([1 for row in data if row['type'] == "CODE"])))
    print("Total {}".format(len(data)))


def write(data, fname):
    with open(fname, 'w') as f:
        json.dump([row for row in data if row["type"] != "COMMENT"], f)


def read(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def main_python():
    data = read('../data/Pairs/code_comment_102813.json')
    print_info(data)
    cleaned = []
    dedup = set()
    cleaned = []
    for row in data:
        key = "{}#{}#{}#{}".format(
            row['before_comment'], row['after_comment'], row['before_path'], row['after_path'])
        if key not in dedup:
            dedup.add(key)
        if not isEnglish(row['before_comment']) or not isEnglish(row['after_comment']):
            continue
        if not row['before_comment'] or not row['after_comment'] or not row['before_code'] or not row['after_code']:
            continue

        if int(row['after_line']) < 10 or int(row['before_line']) < 10:
            continue

        for k in ['before_comment', 'after_comment']:
            comment = clean_comment(row[k])
            row[k] = comment

        for k in ['before_code', 'after_code']:
            code = clean_code(row[k])
            row[k] = code

        if row['before_comment'] != row['after_comment'] and row['before_code'][0] != row['after_code'][0]:
            row['type'] = "BOTH"
        if row['before_comment'] == row['after_comment'] or row['before_code'][0] == row['after_code'][0]:
            row['type'] = "COMMENT"
        cleaned.append(row)
    # write(cleaned, '../data/Pairs/code_comment_10k.json')
    print_info(cleaned)


def main_java(data_path):
    data = read(data_path)
    # lexer = build_lexer('java')

    print_info(data)
    dedup = set()
    cleaned = []
    for row in data:
        key = "{}#{}#{}#{}".format(
            row['before_comment'], row['after_comment'], row['before_path'], row['after_path'])

        if key not in dedup:
            dedup.add(key)
        if not isEnglish(row['before_comment']) or not isEnglish(row['after_comment']):
            continue
        if not isEnglish("".join(row['before_code'])) or not isEnglish("".join(row['after_code'])):
            continue
        if not row['before_comment'] or not row['after_comment'] or not row['before_code'] or not row['after_code']:
            continue

        if int(row['after_line']) < 20 or int(row['before_line']) < 20:
            continue

        both_cond = row['before_comment'] != row['after_comment'] and "".join(
            row['before_code']) != "".join(row['after_code'])

        comment_cond = row['before_comment'] != row['after_comment'] and "".join(
            row['before_code']) == "".join(row['after_code'])
        code_cond = row['before_comment'] == row['after_comment'] and "".join(
            row['before_code']) != "".join(row['after_code'])

        if both_cond:
            row['type'] = "BOTH"
        elif comment_cond:
            row['type'] = "COMMENT"
        elif code_cond:
            row['type'] = "CODE"

        cleaned.append(row)
    write(cleaned, data_path)
    print_info(cleaned)


def merge_python():
    code_changes = read('../data/Pairs/code_changes_6826.json')
    # both_changes = read('../data/Pairs/code_comment_10k.json')
    # lexer = build_lexer('python')

    # for code, both in zip(code_changes, both_changes):
    cleaned = []
    for code in code_changes:
        if code['before_code'] != code['after_code']:
            cleaned.append(code)
    print(len(cleaned))


def pprint_file_name(file_name, row):
    org, project, commit, changed_file = file_name.split("#")
    commit = row['commit']
    changed_file = changed_file.split('__')[-1]
    url = f'https://github.com/{org}/{project}/commit/{commit}'
    line = row['after_line']
    return url, changed_file, line


def split():

    data = read('../data/Pairs/code_comment_738.json')
    both = [row for row in data if row['type'] == "BOTH"]
    code = [row for row in data if row['type'] == "CODE"]

    to_inspect = []
    for row in random.sample(both, 25):
        keep = {
            "id": row['after_path'] + "#" + row['after_line'],
            "before_comment": row['before_comment'].split("\\n"),
            "before_code": row['before_code'],
            "after_code": row['after_code'],
            "label": "1"
        }
        to_inspect.append(keep)

    for row in random.sample(code, 25):
        keep = {
            "id": row['after_path'] + "#" + row['after_line'],
            "before_comment": row['before_comment'].split("\\n"),
            "before_code": row['before_code'],
            "after_code": row['after_code'],
            "label": "0"
        }
        to_inspect.append(keep)

    with open('../data/john.json', 'w+') as f:
        json.dump(to_inspect, f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="Path to training data")
    args = ap.parse_args()
    main_java(args.data)
