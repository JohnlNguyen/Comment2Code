from git_crawler import get_postcommit_file, get_precommit_file, ADDED_MODE, REMOVED_MODE
import csv
import os
import sys

from collections import namedtuple
from pathlib import Path


def path_to_file(org, project):
	return os.path.join("Repos", org, project)


def create_file_name(org, project, file, commit, is_added=False):
	file_name = f'{org}#{project}#{commit}#{file.replace("/", "__")}'
	if is_added:
		return Path(os.path.join("files-post", file_name))
	else:
		return Path(os.path.join("files-pre", file_name))


def extract_added_code(csv_file_name: str, commit_id: str, dir_path: str, file: str):
	with open(csv_file_name, newline='') as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		# using csv header as namedtuple fields
		DataRow = namedtuple('DataRow', next(reader))

		for row in map(DataRow._make, reader):
			if row.mode == ADDED_MODE:
				get_postcommit_file(
					path=path_to_file(row.organization, row.project),
					commit_id=row.commit,
					file=row.file_changed,
					out_file=create_file_name(row.organization, row.project, row.file_changed, row.commit,
											  is_added=True).open('w')
				)
			elif row.mode == REMOVED_MODE:
				get_precommit_file(
					path=path_to_file(row.organization, row.project),
					commit_id=row.commit,
					file=row.file_changed,
					out_file=create_file_name(row.organization, row.project, row.file_changed, row.commit,
											  is_added=False).open('w')
				)


def main(dir_path):
	diff_list = os.listdir(dir_path)
	for csv_diff_file in diff_list:
		extract_added_code(os.path.join(dir_path, csv_diff_file), "", "", "")


if __name__ == '__main__':
	diffs_dir = sys.argv[2] if len(sys.argv) > 2 else 'Diffs'
	main(diffs_dir)
