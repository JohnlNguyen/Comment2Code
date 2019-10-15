import argparse
import csv
import linecache
import os
import pickle
import traceback
import sys
from collections import namedtuple
from pathlib import Path

import pygments
from git_crawler import get_postcommit_file, get_precommit_file, ADDED_MODE, REMOVED_MODE
from lexer import strip_special_chars, build_lexer, lex_content
from pygments.token import Comment, Text, Keyword
from utils import is_a_comment_line, filter_comments, filter_code, is_a_code_line, contains_a_comment

CommentCodeRow = namedtuple(
    'CommentCodeRow', ['comment', 'code', 'heuristic', 'file', 'lineno', 'type'])


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


def path_to_file(org, project):
    return os.path.join(repos_dir, org, project)


def create_file_name(org, project, file, commit, is_added=False):
    file_name = "{}#{}#{}#{}".format(org, project, commit, file.replace("/", "__"))
    if is_added:
        return Path(os.path.join("files-post", file_name))
    else:
        return Path(os.path.join("files-pre", file_name))


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
        code_end = min(code_end + Heuristic.LOOK_AHEAD_LINES, len(to_extract_content))
        code = to_extract_content[:code_end]

        code = lex_content("".join(code), lexer)
        comment = strip_special_chars("".join(comment))
    else:
        code = to_extract_content[:code_end]
        code = lex_content("".join(code), lexer)
        comment = strip_special_chars("".join(comment))

    # comment and code are on the same line case
    if not comment:
        if len(content) < start_lineno:
            print("Length of content is less than start_line {}".format(start_lineno))
            return None, None, None

        ttypes = [t for t, _ in pygments.lex(content[start_lineno], lexer)]
        if is_a_code_line(ttypes) and contains_a_comment(ttypes):
            line = content[start_lineno].split("#")
            code, comment = line[:-1], line[-1]
            code = lex_content("".join(code), lexer)
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


def get_commit_files(csv_file_name: str):
    comment_code_rows = []
    with open(csv_file_name, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        # using csv header as namedtuple fields
        DataRow = namedtuple('DataRow', next(reader))

        for row in map(DataRow._make, reader):
            file_name = create_file_name(row.organization, row.project, row.file_changed, row.commit,
                                         is_added=row.mode == ADDED_MODE)
            if row.mode == ADDED_MODE:
                get_postcommit_file(
                    path=path_to_file(row.organization, row.project),
                    commit_id=row.commit,
                    file=row.file_changed,
                    out_file=file_name.open('w'),
                )
            elif row.mode == REMOVED_MODE:
                get_precommit_file(
                    path=path_to_file(row.organization, row.project),
                    commit_id=row.commit,
                    file=row.file_changed,
                    out_file=file_name.open('w')
                )
            lineno = int(row.comment_line_added if row.mode ==
                         ADDED_MODE else row.comment_line_removed)
            comment, code, heuristic = extract_code(lineno, file_name)

            if comment and code and heuristic:
                comment_code_rows.append(
                    CommentCodeRow(comment=comment, code=code, heuristic=heuristic, file=file_name.as_posix(),
                                   lineno=lineno, type=row.change_type))
    return comment_code_rows


def write_data(rows, file_name):
    with open('Pairs/{}.pkl'.format(file_name), 'wb') as f:
        pickle.dump(rows, f)
    print("Total Comment-Code Pairs written {}".format(len(rows)))
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Extract code from commit files")
    parser.add_argument("-d", "--dir", required=False, default="Diffs",
                        help="Directory to extract code from")
    parser.add_argument("-r", "--repos", required=False, default="Repos")
    return parser.parse_args()


def main(dir_path):
    diff_list = os.listdir(dir_path)
    for csv_diff_file in diff_list:
        print("Extracting code for {}".format(csv_diff_file))
        path = os.path.join(dir_path, csv_diff_file)
        try:
            data_rows = get_commit_files(path)
            data_file_name = os.path.splitext(csv_diff_file)[0]
            write_data(data_rows, data_file_name)
        except Exception as e:
            print("Exception processing", csv_diff_file, "--", traceback.print_exc(file=sys.stdout))


if __name__ == '__main__':
    args = parse_args()
    diffs_dir = args.dir
    repos_dir = args.repos
    main(diffs_dir)
