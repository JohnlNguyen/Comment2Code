import os
import sys
import re
import subprocess
import csv
import shutil
from lexer import lex_file
import whatthepatch
import traceback
from typing import Generator, Iterator
from collections import namedtuple

Metadata = namedtuple('Metadata', ['org', 'project', 'commit_id'])
ADDED_MODE = "ADDED"
REMOVED_MODE = "REMOVED"


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


def parse_line_indices(text):
	text = text.strip()
	if text.startswith("+") or text.startswith("-"): text = text[1:]  # Remove + or -
	if "," in text:
		s = text.split(",")
		start = int(s[0])
		return start, start + int(s[1]) - 1
	else:
		return int(text), int(text)


class Diff():
	def __init__(self, org, project, commit_id):
		self.org = org
		self.project = project
		self.commit_id = commit_id
		self.files_changed = {}

	def add_changed_file(self, file):
		if file in self.files_changed:
			print("Changed file already recorded for commit?", file)
			return
		self.files_changed[file] = []

	def add_changed_lines(self, file, rm_start, rm_end, add_start, add_end):
		if file not in self.files_changed:
			print("Changed file not recorded for commit?", file)
			return
		self.files_changed[file].append((rm_start, rm_end, add_start, add_end))


def parse(diff, dir_path, file_name, out_file):
	valid_diffs = {}
	with open(file_name, "r", encoding="utf8", errors='ignore') as file:
		curr_file = None  # We'll use these variables to confirm we are in a valid diff
		offset_r = 0  # To track the shift in line alignment
		for line in file:
			# First, if this line describes the files changed, assert that the extensions are appropriate (in our case, Python)
			if line.startswith("diff --git"):
				start, end = [m.span() for m in re.finditer(" a/.+ b/", line)][0]
				changed_file = line[start:end][3:-3].strip()
				diff.add_changed_file(changed_file)  # Store the changed file for bookkeeping, even if it is not valid
				if changed_file.endswith(".py"):
					curr_file = changed_file
					offset_r = 0
				else:
					curr_file = None
			if curr_file == None: continue  # Don't bother parsing if we are not in a valid file

			# Next, check that the exact line diff is valid. In our case, this means making sure that we can easily infer alignment (e.g. same line removed and added)
			if line.startswith("@@"):
				parts = line.rstrip().split(" ")
				if parts[1].startswith('+') or parts[2] == '@@': continue  # Skip changes with no additions or deletions
				rm_start, rm_end = parse_line_indices(parts[1])
				add_start, add_end = parse_line_indices(parts[2])
				# To ensure we are capturing alignment, shift line numbers in new file, first in comparison, end then to store new offset
				if rm_end < (add_start + offset_r) or (add_end + offset_r) < rm_start:
					return  # If we find a violation in alignment, exit immediately
				offset_r += (rm_end - rm_start) - (
					add_end - add_start)  # Captures degree to which new file is "ahead" (or behind)
				diff.add_changed_lines(curr_file, rm_start, rm_end, add_start, add_end)

	python_files_changed = len([f for f in diff.files_changed if f.endswith(".py")])
	if python_files_changed == 0: return

	# Append results to data file
	with open(out_file, 'a', newline='') as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for file in diff.files_changed.keys():
			for (rm_start, rm_end, add_start, add_end) in diff.files_changed[file]:
				writer.writerow(
					[diff.org, diff.project, diff.commit_id, len(diff.files_changed), python_files_changed, file,
					 len(diff.files_changed[file]), rm_end - rm_start + 1, add_end - add_start + 1, rm_start, rm_end,
					 add_start, add_end])
			if not file.endswith(".py"): continue
			file_name_pre = "files-pre/" + diff.org + '#' + diff.project + '#' + diff.commit_id + '#' + file.replace(
				"/", "__")
			file_name_post = "files-post/" + diff.org + '#' + diff.project + '#' + diff.commit_id + '#' + file.replace(
				"/", "__")
			with open(file_name_pre, "w") as of:
				get_precommit_file(dir_path, diff.commit_id, file, of)
			lexed = lex_file(file_name_pre, "python")
			with open(file_name_pre, "w") as of:
				for l in lexed:
					of.write("\t".join(l))
					of.write("\n")
			with open(file_name_post, "w") as of:
				get_postcommit_file(dir_path, diff.commit_id, file, of)
			lexed = lex_file(file_name_post, "python")
			with open(file_name_post, "w") as of:
				for l in lexed:
					of.write("\t".join(l))
					of.write("\n")


def main(in_dir, out_dir):
	orgs_list = os.listdir(in_dir)
	# Empty the output file safe for the header; we will append to it for every project
	for org in orgs_list:
		projects_list = os.listdir(os.path.join(in_dir, org))
		for project in projects_list:
			try:
				print("Processing {0}/{1}".format(org, project))
				dir_path = os.path.join(in_dir, org, project)
				out_file = os.path.join(out_dir, '{0}__{1}.csv'.format(org, project))
				# write to csv
				with open(out_file, 'w', encoding='utf8', newline='') as csv_file:
					writer = csv.writer(csv_file, delimiter=',')
					writer.writerow(
						['organization', 'project', 'commit', 'mode', 'file_changed', 'comment_line_removed',
						 'comment_line_added'])

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


def is_valid_change(diff):
	"""
	Criteria for us to accept a diff
		- File must be a python file
		- Diff must start with a comment (might relax this later)

	:param diff: Changes in one file
	:return:
	"""
	if not diff.header.new_path.endswith(".py") or not diff.header.old_path.endswith(".py"):
		return False

	if not diff.changes:
		return False

	for change in diff.changes:
		striped_line = change.line.strip()
		# TODO: only accept commits starting with comment
		if striped_line and not striped_line.startswith("#"):
			return False

	return True


import itertools


def parse_diff(diffs: Generator, meta: Metadata, csv_out_file):
	for diff in diffs:
		if not is_valid_change(diff):
			continue

		# filter out empty line changes
		changes = [x for x in diff.changes if x.line]

		# ignoring line shift changes
		added_changes = [x for x in changes if not bool(x.old) and bool(x.new)]
		removed_changes = [x for x in changes if not bool(x.new) and bool(x.old)]

		if diff.header.old_path != diff.header.new_path:
			print("{}:Paths are different Old:{} New:{}".format(meta, diff.header.old_path, diff.header.new_path))
			continue

		removed_changes = remove_consecutive(removed_changes, is_added=False)
		added_changes = remove_consecutive(added_changes, is_added=True)
		write_to_csv(added_changes + removed_changes, csv_out_file, diff.header.new_path, meta)
	return


def remove_consecutive(changes, is_added=True):
	filtered_changes = []
	if not changes:
		return filtered_changes

	if len(changes) == 1:
		filtered_changes.extend(changes)
	else:
		filtered_changes.append(changes[0])
		for i in range(1, len(changes)):
			if is_added and (changes[i].new - changes[i - 1].new) == 1:
				continue
			if not is_added and (changes[i].old - changes[i - 1].old) == 1:
				continue

			filtered_changes.append(changes[i])
	return filtered_changes


def write_to_csv(changes: Iterator, csv_file_name: str, file_changed: str, meta: Metadata):
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
			row = [meta.org, meta.project, meta.commit_id, mode, file_changed, change.old, change.new]
			writer.writerow(row)


if __name__ == '__main__':
	in_dir = sys.argv[1] if len(sys.argv) > 1 else 'Repos'
	out_dir = sys.argv[2] if len(sys.argv) > 2 else 'Diffs'
	main(in_dir, out_dir)
