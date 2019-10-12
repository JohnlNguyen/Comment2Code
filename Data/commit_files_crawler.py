import argparse
import csv
import linecache
import os
from collections import namedtuple
from pathlib import Path

from git_crawler import get_postcommit_file, get_precommit_file, ADDED_MODE, REMOVED_MODE

import pygments
from pygments.token import Comment, Text, Name, Keyword

from lexer import strip_special_chars, build_lexer, lex_content
import pickle

DEBUG = False

CommentCodeRow = namedtuple('CommentCodeRow', ['comment', 'code', 'heuristic', 'file', 'lineno'])


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
	return os.path.join("Repos", org, project)


def create_file_name(org, project, file, commit, is_added=False):
	file_name = "{}#{}#{}#{}".format(org, project, commit, file.replace("/", "__"))
	if is_added:
		return Path(os.path.join("files-post", file_name))
	else:
		return Path(os.path.join("files-pre", file_name))


def is_a_comment_line(ttypes):
	return {Name, Keyword} not in ttypes and any([t in Comment for t in ttypes])


def extract_code(start_lineno, file_name):
	content = linecache.getlines(file_name.as_posix())
	lexer = build_lexer('python')

	# look backward to capture the entire comment in case we are the middle of a multiline comment
	for line in reversed(content[:start_lineno]):
		tokens = list(pygments.lex(line, lexer))
		# if a line has no keyword, name token
		# and has a comment token we assume it is a part of the initial comment
		ttypes, _ = list(zip(*tokens))
		if is_a_comment_line(ttypes):
			start_lineno -= 1
		else:
			break

	to_extract_content = content[start_lineno:]
	comment_len = 0
	code_end = 1

	heuristic = None
	for i, line in enumerate(to_extract_content):
		tokens = list(pygments.lex(line, lexer))
		# if a line has no keyword, name token
		# and has a comment token we assume it is a part of the initial comment
		ttypes, _ = list(zip(*tokens))
		if is_a_comment_line(ttypes):
			comment_len += 1
			continue

		should_stop, reason = Heuristic.should_stop(tokens)
		if should_stop:
			heuristic = reason
			code_end = i
			break

	if heuristic == Heuristic.BLANK_LINE:
		comment = to_extract_content[:comment_len]

		# get the next 10 lines following the blank line
		code = to_extract_content[comment_len: min(code_end + 11, len(to_extract_content))]
		code = lex_content("".join(code), lexer)
		comment = strip_special_chars("".join(comment))
		if DEBUG:
			print(comment, code)
		return comment, code, heuristic

	comment = to_extract_content[:comment_len]
	code = to_extract_content[comment_len:code_end + 1]
	code = lex_content("".join(code), lexer)
	comment = strip_special_chars("".join(comment))
	if DEBUG:
		print(comment, code)
	return comment, code, heuristic


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
			lineno = int(row.comment_line_added if row.mode == ADDED_MODE else row.comment_line_removed)
			comment, code, heuristic = extract_code(lineno, file_name)
			comment_code_rows.append(
				CommentCodeRow(comment=comment, code=code, heuristic=heuristic, file=file_name.as_posix(),
							   lineno=lineno))
	return comment_code_rows


def write_data(rows):
	with open('comment_code.pkl', 'wb') as f:
		pickle.dump(rows, f)
	print("Total Comment-Code Pairs written {}".format(len(rows)))
	return


def parse_args():
	parser = argparse.ArgumentParser(description="Extract code from commit files")
	parser.add_argument("-d", "--dir", required=False, default="Diffs",
						help="Directory to extract code from")
	parser.add_argument("-t", "--test", required=False, default=False,
						help="Test mode or not")
	parser.add_argument("-db", "--debug", required=False, default=False,
						help="Debug or not")
	return parser.parse_args()


def main(dir_path):
	diff_list = os.listdir(dir_path)
	data = []
	for csv_diff_file in diff_list:
		data.extend(get_commit_files(os.path.join(dir_path, csv_diff_file)))

	write_data(data)


if __name__ == '__main__':
	args = parse_args()
	diffs_dir = args.dir
	DEBUG = args.debug
	if args.test:
		print("".join(linecache.getlines('../test/t.py')))
		extract_code(2, Path('../test/t.py'))
	else:
		main(diffs_dir)
