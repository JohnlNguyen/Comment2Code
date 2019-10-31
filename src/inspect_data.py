from pathlib import Path
import os
import pprint
import argparse
import random
import json
import pdb


def pprint_file_name(file_name):
    org, project, commit, changed_file = file_name.split("#")
    return org, project, commit, changed_file


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect")
    parser.add_argument("-f", "--file", required=False, default='comment_code.json',
                        help="File name")
    return parser.parse_args()


def calc_acc(ans, data):
    correct = 0
    for a, e in zip(ans, [row['label'] for row in data]):
        if a == e:
            correct += 1
    return correct / len(data)

if __name__ == '__main__':
    args = parse_args()
    pp = pprint.PrettyPrinter(indent=2)
    with open(Path('../data/' + args.file).as_posix(), 'rb') as f:
        data = json.load(f)

    ans = []
    for i, row in enumerate(data):
        print("-" * 50)
        print("Comment")
        print(row['comment'])
        print("Code")
        pp.pprint(row['code'])

        query = input("Press enter to 1 for aligned 0 for not aligned or type q to quit: ")
        if query == 'q':
            exit(0)
        ans.append(int(query))

    print("Accuracy {}".format(calc_acc(ans, data)))
