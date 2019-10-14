import argparse
import csv
import os
import subprocess
import sys
import traceback
from collections import namedtuple

import pygments
import whatthepatch
from lexer import build_lexer
from more_itertools import consecutive_groups
from pygments.token import Name, Keyword, Operator
from utils import is_a_comment_line, is_a_code_line, contains_a_comment

Metadata = namedtuple('Metadata', ['org', 'project', 'commit_id'])
ADDED_MODE = "ADDED"
REMOVED_MODE = "REMOVED"


class GitChange(object):
	BOTH = "BOTH"
	COMMENT = "COMMENT"
	CODE = "CODE"

	def __init__(self, old, new, line, ctype):
		self.old = old
		self.new = new
		self.line = line
		self.type = ctype

	def __str__(self):
		return "old={},new={},line={},type={}".format(self.old, self.new, self.line, self.type)

	def __repr__(self):
		return self.__str__()


class CrawlMode(object):
	STARTS_WITH_COMMENT = "STARTS_WITH_COMMENT"
	COMMENT_IN_DIFF = "COMMENT_IN_DIFF"

	def __init__(self, m):
		self.mode = m

	def is_valid_diff(self, diff):
		"""
		Criteria for us to accept a diff
			- File must be a python file
			- Diff must has a comment

		:param diff: Changes in one file
		:return:
		"""
		if not diff.header.new_path.endswith(".py") or not diff.header.old_path.endswith(".py"):
			return False

		if not diff.changes:
			return False

		lines = [change.line.strip() for change in diff.changes]
		if self.mode == CrawlMode.COMMENT_IN_DIFF:
			if not any([line.startswith("#") for line in lines]):
				return False
		elif self.mode == CrawlMode.STARTS_WITH_COMMENT:
			for line in lines:
				if line and not line.startswith("#"):
					return False

		return True


mode = CrawlMode(CrawlMode.STARTS_WITH_COMMENT)


def get_git_revision_hash(path):
	return subprocess.check_output(['git', '-C', path, 'rev-parse', '--abbrev-ref', 'HEAD']).decode("utf-8").rstrip(
		"\n")


def get_entire_history(path, curr_branch):
	return subprocess.check_output([
		'git', '-C', path, 'rev-list', curr_branch, '--first-parent', '--abbrev-commit'
	]).decode("utf-8").rstrip("\n").split("\n")


# Compute diff of commit ID, relative to previous commit if one exists
def get_diff(path, commit_id, out_file, relative_to_parent=True):
	if relative_to_parent:
		return subprocess.call(['git', '-C', path, 'diff', commit_id + '^1', commit_id, '-U0'], stdout=out_file)
	else:
		return subprocess.call(['git', '-C', path, 'diff', commit_id, '-U0'], stdout=out_file)


def get_precommit_file(path, commit_id, file, out_file):
	return subprocess.call(['git', '-C', path, 'show', commit_id + '~1:' + file], stdout=out_file)


def get_postcommit_file(path, commit_id, file, out_file):
	return subprocess.call(['git', '-C', path, 'show', commit_id + ':' + file], stdout=out_file)


def tag_change(all_changes, lexer, is_added=True):
	changes = []
	for group in consecutive_groups(all_changes, lambda c: c.new if is_added else c.old):
		group = list(group)

		assert len(group) > 0

		change_to_keep = None
		for i, change in enumerate(group):
			ttypes = [t for t, _ in pygments.lex(change.line, lexer)]

			if not change_to_keep and contains_a_comment(ttypes):
				change_to_keep = GitChange(old=change.old, new=change.new, line=change.line, ctype=GitChange.COMMENT)
			if change_to_keep and is_a_code_line(ttypes):
				change_to_keep.type = GitChange.BOTH

		if change_to_keep:
			changes.append(change_to_keep)
	return changes


def parse_diff(diffs, meta, csv_out_file):
	lexer = build_lexer('python')
	for diff in diffs:
		if not mode.is_valid_diff(diff):
			continue

		# filter out empty line changes
		changes = [c for c in diff.changes if c.line]
		# ignoring line shift changes
		added_changes = [x for x in changes if not bool(x.old) and bool(x.new)]
		removed_changes = [x for x in changes if not bool(x.new) and bool(x.old)]

		added_changes = tag_change(added_changes, lexer, is_added=True)
		removed_changes = tag_change(removed_changes, lexer, is_added=False)

		if diff.header.old_path != diff.header.new_path:
			print("{}: Paths are different Old: {} New: {}".format(meta, diff.header.old_path, diff.header.new_path))
			continue

		write_to_csv(added_changes + removed_changes, csv_out_file, diff.header.new_path, meta)
	return


def write_to_csv(changes, csv_file_name: str, file_changed: str, meta):
	if not changes:
		return
	dups = set()
	with open(csv_file_name, 'a', newline='') as csv_file:
		writer = csv.writer(csv_file, delimiter=',')

		for change in changes:
			mode = ADDED_MODE if change.new else REMOVED_MODE
			lineno = change.new if mode == ADDED_MODE else change.old

			# remove duplicates
			change_id = "{}#{}#{}".format(meta.commit_id, file_changed, lineno)
			if change_id in dups:
				continue

			dups.add(change_id)
			row = [meta.org, meta.project, meta.commit_id, mode, file_changed, change.old, change.new, change.type]
			writer.writerow(row)


def write_csv_header(out_dir, org, project):
	out_file = os.path.join(out_dir, '{0}__{1}.csv'.format(org, project))
	with open(out_file, 'w', encoding='utf8', newline='') as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		writer.writerow(
			['organization', 'project', 'commit', 'mode', 'file_changed', 'comment_line_removed',
			 'comment_line_added', 'change_type'])
	return out_file


def main(in_dir, out_dir):
	orgs_list = os.listdir(in_dir)
	# Empty the output file safe for the header; we will append to it for every project
	for org in orgs_list:
		projects_list = os.listdir(os.path.join(in_dir, org))
		for project in projects_list:
			try:
				print("Processing {0}/{1}".format(org, project))
				dir_path = os.path.join(in_dir, org, project)
				out_file = write_csv_header(out_dir, org, project)

				curr_branch = get_git_revision_hash(dir_path)
				all_commit_ids = get_entire_history(dir_path, curr_branch)
				for ix, commit_id in enumerate(all_commit_ids):
					with open("output.diff", "w", encoding="utf8") as of:
						is_last = ix == len(all_commit_ids) - 1
						get_diff(dir_path, commit_id, of, relative_to_parent=not is_last)

					# parsing
					try:
						with open("output.diff", "r", encoding="utf8", errors='ignore') as f:
							text = f.read()
						parse_diff(whatthepatch.parse_patch(text),
								   Metadata(org=org, project=project, commit_id=commit_id), out_file)
					except Exception as e:
						print("Exception parsing diff", org, project, commit_id, "--",
							  traceback.print_exc(file=sys.stdout))
			except Exception as e:
				print("Exception processing project", org, project, "--", traceback.print_exc(file=sys.stdout))


def parse_args():
	parser = argparse.ArgumentParser(description="Crawl Git repos")
	parser.add_argument("-i", "--in_dir", required=False, default="Repos",
						help="Directory to write to")
	parser.add_argument("-o", "--out_dir", required=False, default="Diffs",
						help="Repos to crawl through")
	parser.add_argument("-m", "--mode", required=False, default=CrawlMode.COMMENT_IN_DIFF,
						help="Mode to crawl")
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	in_dir = args.in_dir
	out_dir = args.out_dir
	mode.mode = args.mode
	main(in_dir, out_dir)
