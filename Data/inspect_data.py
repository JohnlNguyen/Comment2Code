import pickle
from commit_files_crawler import CommentCodeRow
from pathlib import Path
import os
import pprint


def pprint_file_name(file_name):
	org, project, commit, changed_file = file_name.split("/")[1].split("#")
	print(org, project, commit, changed_file)
	return org, project, commit


if __name__ == '__main__':
	pp = pprint.PrettyPrinter(indent=2)
	with open(Path(os.getcwd(), 'comment_code.pkl').as_posix(), 'rb') as f:
		data = pickle.load(f)
		for row in data:
			query = input("Press enter to inspect next pair or type q to quit: ")
			if query == 'q':
				exit(0)
			print("-"*30)
			print("---Comment---")
			print(row.comment.replace("\\n", "\n").replace("\\t", "\t"))
			print("---Code---")
			pp.pprint(row.code)
			org, proj, commit = pprint_file_name(row.file)
			print(f'Heuristic: {row.heuristic} Line #: {row.lineno}')
			print("Link to commit: https://github.com/{}/{}/commit/{}".format(org, proj, commit))
