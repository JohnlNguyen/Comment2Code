from pathlib import Path
import os
import pprint
import argparse
import random
import pandas as pd
import json
import pdb


def pprint_file_name(file_name):
    org, project, commit, changed_file = file_name.split("#")
    print("Org: {} Project: {} Commit: {} File: {}".format(org, project, commit, changed_file))
    return org, project, commit


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect")
    parser.add_argument("-r", "--rand", required=False, default=False,
                        help="Randomly shuffle when inspect")
    parser.add_argument("-f", "--file", required=False, default='comment_code.json',
                        help="File name")
    return parser.parse_args()


def print_info(data):
    df = pd.DataFrame(data)
    print("Total Comment-Code pairs: {}".format(df.shape[0]))

    for key in ['type', 'before_heuristic', 'after_heuristic']:
        print("{} count {}".format(key, df[key].value_counts()))
    return df

if __name__ == '__main__':
    args = parse_args()
    pp = pprint.PrettyPrinter(indent=2)
    with open(Path('../data/' + args.file).as_posix(), 'rb') as f:
        data = json.load(f)
        data = print_info(data)

        if args.rand:
            random.shuffle(data)

        for _, row in data.iterrows():
            query = input("Press enter to inspect next pair or type q to quit: ")
            if query == 'q':
                exit(0)
            print("-" * 50)
            print("---Before Comment---")
            print(row.before_comment)
            print("---After Comment---")
            print(row.after_comment)
            print("---Before Code---")
            pp.pprint(row.before_code)
            print("---After Code---")
            pp.pprint(row.after_code)
            org, proj, commit = pprint_file_name(row.after_path)
            print("Link to commit: https://github.com/{}/{}/commit/{}".format(org, proj, commit))
