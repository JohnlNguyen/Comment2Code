import argparse
import csv
import linecache
import os
import pickle
import traceback
import sys
import pdb
import subprocess
import ray
import itertools
import pygments
import json

from collections import namedtuple
from pathlib import Path


from git_crawler import get_postcommit_file, get_precommit_file, ADDED_MODE, REMOVED_MODE
from lexer import strip_special_chars, build_lexer, lex_content
from pygments.token import Comment, Text, Keyword
from utils import is_a_comment_line, filter_comments, filter_code, is_a_code_line, contains_a_comment

CommentCodeRow = namedtuple(
    'CommentCodeRow', ['before_comment', 'before_code', 'before_heuristic', 'after_comment', 'after_code', 'after_heuristic',
                       'before_path', 'after_path', 'before_line', 'after_line', 'type'])

DATA_DIR = "../data"


class Heuristic(object):
    """
    - Get code up to the next blank line (storing extra 10 lines ahead)
    - Get code till the end of the function (return statement)
    - Get code if code and comment are on the same line
    - Get code up to the next comment
    """
    BLANK_LINE = "BLANK_LINE"
    END_OF_FUNCTION = "END_OF_FUNCTION"
    SAME_LINE = "SAME_LINE"
    NEXT_COMMENT = "NEXT_COMMENT"

    LOOK_AHEAD_LINES = 11  # look ahead 10 lines

    @classmethod
    def should_stop(cls, tokens):
        # new line heuristic
        if len(tokens) == 1 and tokens[0][0] == Text and tokens[0][1] == "\n":
            return True, Heuristic.BLANK_LINE

        for ttype, token in tokens:
            # end of function heuristic
            if ttype == Keyword and token == 'return':
                return True, Heuristic.END_OF_FUNCTION

            # next comment heuristic
            if ttype in Comment:
                return True, Heuristic.NEXT_COMMENT
        return False, None


def path_to_file(repo_dir, org, project):
    return os.path.join(repo_dir, org, project)


def create_file_name(org, project, file, commit, is_added=False):
    file_name = "{}#{}#{}#{}".format(
        org, project, commit, file.replace("/", "__"))
    if is_added:
        return Path(os.path.join(DATA_DIR, "files-post", file_name)), file_name
    else:
        return Path(os.path.join(DATA_DIR, "files-pre", file_name)), file_name


def extract_code(start_lineno, file_name):
    content = linecache.getlines(file_name.as_posix())
    lexer = build_lexer('python')

    # content array is 0 index so need to shift down by 1
    start_lineno = max(0, start_lineno - 1)
    comment, comment_end = capture_comment(content, lexer, start_lineno)

    to_extract_content = content[comment_end:]
    code_end = 1
    heuristic = None

    for i, line in enumerate(to_extract_content):
        tokens = list(pygments.lex(line, lexer))

        should_stop, reason = Heuristic.should_stop(tokens)
        if should_stop:
            heuristic = reason
            code_end = i
            break

    if heuristic == Heuristic.BLANK_LINE:
        code_end = min(code_end + Heuristic.LOOK_AHEAD_LINES,
                       len(to_extract_content))
        code = to_extract_content[:code_end]
        comment = strip_special_chars("".join(comment))
    else:
        code = to_extract_content[:code_end]
        comment = strip_special_chars("".join(comment))

    # comment and code are on the same line case
    if not comment:
        if len(content) - 1 < start_lineno:
            print("Length of content is less than start_line {}".format(
                file_name.as_posix()))
            return None, None, None

        ttypes = [t for t, _ in pygments.lex(content[start_lineno], lexer)]
        if is_a_code_line(ttypes) and contains_a_comment(ttypes):
            line = content[start_lineno].split("#")
            code, comment = line[:-1], line[-1]
            comment = strip_special_chars("".join(comment))

    return comment, code, heuristic


def capture_comment(content, lexer, start):
    # look backward to capture the entire comment in case we are the middle of a multiline comment
    comment_start = comment_end = start
    for line in reversed(content[:start]):
        ttypes = [t for t, _ in pygments.lex(line, lexer)]
        # if a line has no keyword, name or operator
        # and has a comment token we assume it is a part of the initial comment
        if is_a_comment_line(ttypes):
            comment_start -= 1
        else:
            break

    # look forward to capture the entire comment in case we are the middle of a multiline comment
    for line in content[start:]:
        ttypes = [t for t, _ in pygments.lex(line, lexer)]
        if is_a_comment_line(ttypes):
            comment_end += 1
        else:
            break
    return content[comment_start:comment_end], comment_end


def get_commit_file(path, commit_id, out_file):
    return subprocess.run(['git', 'show', commit_id], cwd=path, stdout=out_file)


@ray.remote
def get_commit_files_and_extract(csv_file_name, repo_dir):
    comment_code_rows = []

    try:
        print("Extracting code for {}".format(csv_file_name))
        with open(csv_file_name, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            # using csv header as namedtuple fields
            DataRow = namedtuple('DataRow', next(reader))

            for row in map(DataRow._make, reader):
                full_bf_name, bf_name = create_file_name(
                    row.org, row.project, row.before_path, row.before_commit, is_added=False)

                full_af_name, af_name = create_file_name(
                    row.org, row.project, row.after_path, row.after_commit, is_added=True)

                get_commit_file(
                    path=path_to_file(repo_dir, row.org, row.project),
                    commit_id=row.before_commit,
                    out_file=full_af_name.open('w'),
                ),
                get_commit_file(
                    path=path_to_file(repo_dir, row.org, row.project),
                    commit_id=row.after_commit,
                    out_file=full_bf_name.open('w')
                )

                # lineno = int(row.comment_line_added if row.mode == ADDED_MODE else row.comment_line_removed)
                bf_comment, bf_code, bf_heuristic = extract_code(
                    int(row.added_line), full_af_name)

                af_comment, af_code, af_heuristic = extract_code(
                    int(row.rm_line), full_bf_name)

                comment_code_rows.append(CommentCodeRow(bf_comment, bf_code, bf_heuristic, af_comment, af_code,
                                                        af_heuristic, bf_name, af_name, row.added_line, row.rm_line, row.change_type)._asdict())
    except Exception as e:
        print("Exception processing", csv_file_name, "--", traceback.print_exc(file=sys.stdout))

    return comment_code_rows

def write_data(rows, file_name):
    if not rows:
        return 0

    with open(DATA_DIR + '/Pairs/{}.json'.format(file_name), 'w+') as f:
        json.dump(rows, f)
    print("Comment-Code Pairs written {} for {}".format(len(rows), file_name))
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract code from commit files")
    parser.add_argument("-d", "--dir", required=False, default="../data/Diffs",
                        help="Directory to extract code from")
    parser.add_argument("-r", "--repos", required=False,
                        default="../data/Repos")
    return parser.parse_args()


def main(dir_path, repo_dir):
    diff_list = os.listdir(dir_path)
    total = 0

    result_ids = []
    for idx, csv_diff_file in enumerate(diff_list):
        path = os.path.join(dir_path, csv_diff_file)
        result_ids.append(get_commit_files_and_extract.remote(path, repo_dir))

    results = list(itertools.chain.from_iterable(ray.get(result_ids)))
    write_data(results, 'code_comment_{}'.format(len(results)))
    total += len(results)

if __name__ == '__main__':
    args = parse_args()
    diffs_dir = args.dir
    repos_dir = args.repos
    ray.init(num_cpus=os.cpu_count() // 2)
    import time
    s = time.perf_counter()
    main(diffs_dir, repos_dir)
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
