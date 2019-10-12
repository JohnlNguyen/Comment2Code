import pickle
from commit_files_crawler import CommentCodeRow
from pathlib import Path
import os
import pprint
import argparse
import random


def pprint_file_name(file_name):
	org, project, commit, changed_file = file_name.split("/")[1].split("#")
	print("Org {} Project {} Commit {} File {}".format(org, project, commit, changed_file))
	return org, project, commit


def parse_args():
	parser = argparse.ArgumentParser(description="Inspect")
	parser.add_argument("-r", "--rand", required=False, default=False,
						help="Randomly shuffle when inspect")
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	pp = pprint.PrettyPrinter(indent=2)
	with open(Path(os.getcwd(), 'comment_code.pkl').as_posix(), 'rb') as f:
		data = pickle.load(f)
		print("Total Comment-Code pairs: {}".format(len(data)))

		if args.rand:
			random.shuffle(data)

		for row in data:
			query = input("Press enter to inspect next pair or type q to quit: ")
			if query == 'q':
				exit(0)
			print("-" * 30)
			print("---Comment---")
			print(row.comment.replace("\\n", "\n").replace("\\t", "\t"))
			print("---Code---")
			pp.pprint(row.code)
			org, proj, commit = pprint_file_name(row.file)
			print("Heuristic: {} Line #: {}".format(row.heuristic, row.lineno))
			print("Link to commit: https://github.com/{}/{}/commit/{}".format(org, proj, commit))
