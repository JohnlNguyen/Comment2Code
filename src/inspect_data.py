import pickle
from code_crawler import CommentCodeRow
from pathlib import Path
import os
import pprint
import argparse
import random
import pandas as pd


def pprint_file_name(file_name):
	org, project, commit, changed_file = file_name.split("#")
	print("Org: {} Project: {} Commit: {} File: {}".format(org, project, commit, changed_file))
	return org, project, commit


def parse_args():
	parser = argparse.ArgumentParser(description="Inspect")
	parser.add_argument("-r", "--rand", required=False, default=False,
						help="Randomly shuffle when inspect")
	parser.add_argument("-f", "--file", required=False, default='comment_code.pkl',
						help="File name")
	return parser.parse_args()


def print_info(data):
	df = pd.DataFrame(data)
	print("Total Comment-Code pairs: {}".format(df.shape[0]))
	print("Type Count")
	print(df['type'].value_counts())
	print(df['heuristic'].value_counts())


if __name__ == '__main__':
	args = parse_args()
	pp = pprint.PrettyPrinter(indent=2)
	with open(Path('../data/' + args.file).as_posix(), 'rb') as f:
		data = pickle.load(f)
		print_info(data)

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
			print("Type: {} Heuristic: {} Line #: {}".format(row.type, row.heuristic, row.lineno))
			print("Link to commit: https://github.com/{}/{}/commit/{}".format(org, proj, commit))
