import unittest
import csv
import os

from pathlib import Path


def test_crawler():
	out_dir = "./Diffs"
	out_file = os.path.join(out_dir, "johnlnguyen__banner.csv")
	expect_file = os.path.join(out_dir, "git_crawler_expect.csv")

	assert Path(out_file).exists()
	actual = {}
	with open(out_file, newline='') as csv_file:
		reader = csv.DictReader(csv_file, delimiter=',')
		actual = [row for row in reader]
	expect = {}
	with open(expect_file, newline='') as csv_file:
		reader = csv.DictReader(csv_file, delimiter=',')
		expect = [row for row in reader]

	for a, e in zip(actual, expect):
		assert a['mode'] == e['mode']
		if a['mode'] == "ADDED":
			assert a['comment_line_added'] == e['comment_line_added']
		else:
			assert a['comment_line_removed'] == e['comment_line_removed']
	print('Test passed')

def run_tests():
	test_crawler()


if __name__ == '__main__':
	run_tests()
