import argparse
import csv
import os
import subprocess
import sys
import ray
import traceback
from collections import namedtuple
from itertools import groupby

import pygments
import whatthepatch
from lexer import build_lexer
from more_itertools import consecutive_groups
from utils import is_a_code_line, contains_a_comment, is_a_comment_line_java, is_a_comment_line_python
from pdb import set_trace

Metadata = namedtuple('Metadata', ['org', 'project', 'commit'])
ADDED_MODE = "ADDED"
REMOVED_MODE = "REMOVED"
dups = set()


class GitChange(object):
    BOTH = "BOTH"
    COMMENT = "COMMENT"
    CODE = "CODE"

    def __init__(self, old, new, commit_id, line, ctype):
        self.old = old
        self.new = new
        self.commit_id = commit_id
        self.line = line
        self.type = ctype

    def __str__(self):
        return "commit={},old={},new={},line={},type={}".format(self.commit_id, self.old, self.new, self.line,
                                                                self.type)

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
        # if not diff.header.new_path.endswith(".java") or not diff.header.old_path.endswith(".java"):
        if not diff.header.new_path.endswith(".py") or not diff.header.old_path.endswith(".py"):
            return False

        if not diff.changes:
            return False

        lines = [change.line.strip() for change in diff.changes]
        if self.mode == CrawlMode.COMMENT_IN_DIFF:
            # if not any([line.startswith("//") for line in lines]):
            if not any([line.startswith("#") for line in lines]):
                return False

        return True


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
        return subprocess.call(['git', '-C', path, 'diff', commit_id + '^1', commit_id, '-U10'], stdout=out_file)
    else:
        return subprocess.call(['git', '-C', path, 'diff', commit_id, '-U10'], stdout=out_file)


def get_precommit_file(path, commit_id, file, out_file):
    return subprocess.call(['git', '-C', path, 'show', commit_id + '~1:' + file], stdout=out_file)


def get_postcommit_file(path, commit_id, file, out_file):
    return subprocess.call(['git', '-C', path, 'show', commit_id + ':' + file], stdout=out_file)


def tag_change(all_changes, lexer, meta, is_added=True):
    changes = []
    for group in consecutive_groups(all_changes, lambda c: c.new if is_added else c.old):
        group = list(group)

        assert len(group) > 0
        change_to_keep = None
        for change in group:
            ttypes = [t for t, _ in pygments.lex(change.line, lexer)]
            commit_id = meta.new_version if is_added else meta.old_version
            # skipping over same line comments
            # if not change_to_keep and is_a_comment_line_java(ttypes):
            if not change_to_keep and is_a_comment_line_python(ttypes):
                change_to_keep = GitChange(old=change.old, new=change.new, commit_id=commit_id, line=change.line,
                                           ctype=GitChange.COMMENT)

            if change_to_keep and is_a_code_line(ttypes):
                change_to_keep.type = GitChange.BOTH
            # if change_to_keep and change.old == change.new and is_a_comment_line_java(ttypes):
            if change_to_keep and change.old == change.new and is_a_comment_line_python(ttypes):
                change_to_keep.type = GitChange.CODE

        if change_to_keep:
            changes.append(change_to_keep)
    return changes


mode = CrawlMode(CrawlMode.COMMENT_IN_DIFF)


def parse_diff(diffs, meta, csv_out_file):
    # lexer = build_lexer('java')
    lexer = build_lexer('python')

    total = 0
    for diff in diffs:
        if not mode.is_valid_diff(diff):
            continue

        # filter out empty line changes, and group changes by hunk
        changes = groupby([c for c in diff.changes if c.line],
                          key=lambda x: x.hunk)
        changes = [list(hunk) for _, hunk in changes]

        for hunk in changes:
            # skipping over same line comments
            # comment_lines = [
            #     h for h in hunk if h.line.strip().startswith("//")]
            comment_lines = [h for h in hunk if h.line.strip().startswith("#")]

            if not comment_lines:
                continue
            unchanged_comment_lines = [
                h for h in comment_lines if h.new == h.old]

            # separate remove from added
            added_changes = [
                x for x in hunk if not bool(x.old) and bool(x.new)]
            removed_changes = [
                x for x in hunk if not bool(x.new) and bool(x.old)]

            added_changes = tag_change(
                added_changes, lexer, diff.header, is_added=True)
            removed_changes = tag_change(
                removed_changes, lexer, diff.header, is_added=False)

            # work around to get before and after commit
            unchanged_before = tag_change(
                unchanged_comment_lines, lexer, diff.header, is_added=False)
            unchanged_after = tag_change(
                unchanged_comment_lines, lexer, diff.header, is_added=True)
            unchanged = group_changes(unchanged_before, unchanged_after)

            groups = group_changes(removed_changes, added_changes)
            groups.extend(unchanged)
            if not groups:
                continue

            total += write_to_csv(groups, csv_out_file, diff.header.old_path,
                                  diff.header.new_path, meta)
    return total


def group_changes(removed, added):
    # group together as aligned tuples
    return [(r, a) for r, a in zip(removed, added) if r.old == a.new]


def write_to_csv(changes, csv_file_name, old_path, new_path, meta):
    dups = set()
    with open(csv_file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for before, after in changes:
            # remove duplicates
            change_id = "{}#{}#{}#{}".format(
                before.commit_id, after.commit_id, old_path, new_path)
            # skipping all comment only changes
            if change_id in dups or after.type == GitChange.COMMENT:
                continue
            dups.add(change_id)

            row = [meta.org, meta.project, before.commit_id, old_path,
                   after.commit_id, new_path, before.old, after.new, after.type, meta.commit]
            writer.writerow(row)
    return len(dups)


def write_csv_header(out_dir, org, project):
    out_file = os.path.join(out_dir, '{0}__{1}.csv'.format(org, project))
    with open(out_file, 'w', encoding='utf8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(
            ['org', 'project', 'before_commit', 'before_path',
             'after_commit', 'after_path', 'rm_line', 'added_line', 'change_type', 'commit'])
    return out_file


@ray.remote
def maybe_write_diff(in_dir, org, project):
    try:
        print("Processing {0}/{1}".format(org, project))
        dir_path = os.path.join(in_dir, org, project)
        curr_branch = get_git_revision_hash(dir_path)

        # all commit ids from newest to oldest
        all_commit_ids = get_entire_history(dir_path, curr_branch)

        # iterate from newest to oldest version
        for ix, commit_id in enumerate(all_commit_ids):
            fname = "../data/out-diffs/{}-{}-{}.diff".format(
                org, project, commit_id)
            with open(fname, "w", encoding='iso-8859-1') as of:
                is_last = ix == len(all_commit_ids) - 1
                get_diff(dir_path, commit_id, of,
                         relative_to_parent=not is_last)

        print("Finished writing diff for {0}/{1}".format(org, project))
    except Exception as e:
        print("Exception processing project", org, project,
              "--", traceback.print_exc(file=sys.stdout))


def write_diffs(in_dir):
    orgs_list = os.listdir(in_dir)
    # orgs_list = ["elastic", "spring-projects", "ReactiveX", "square", "apache"]
    result_ids = []
    # Empty the output file safe for the header; we will append to it for every project
    for org in orgs_list:
        projects_list = os.listdir(os.path.join(in_dir, org))
        for project in projects_list:
            out_file = os.path.join(
                out_dir, '{0}__{1}.csv'.format(org, project))
            if os.path.exists(out_file):
                print("File exists {}".format(out_file))
                continue
            result_ids.append(maybe_write_diff.remote(in_dir, org, project))

    ray.get(result_ids)


@ray.remote
def maybe_parse_diff(commit_id, org, out_file, project):
    fname = "../data/out-diffs/{}-{}-{}.diff".format(org, project, commit_id)
    total = 0
    try:
        with open(fname, "r", encoding='iso-8859-1', errors='ignore') as f:
            text = f.read()

        total += parse_diff(whatthepatch.parse_patch(text),
                            Metadata(org=org, project=project, commit=commit_id), out_file)
        print("Finished parsing {}".format(out_file))
    except Exception as e:
        print("Exception parsing diff", org, project, commit_id,
              "--", traceback.print_exc(file=sys.stdout))
    return total


def parse(in_dir, out_dir):
    orgs_list = os.listdir(in_dir)
    result_ids = []
    for org in orgs_list:
        projects_list = os.listdir(os.path.join(in_dir, org))
        for project in projects_list:
            out_file = os.path.join(
                out_dir, '{0}__{1}.csv'.format(org, project))
            if os.path.exists(out_file):
                print("File exists {}".format(out_file))
                continue

            dir_path = os.path.join(in_dir, org, project)
            out_file = write_csv_header(out_dir, org, project)
            curr_branch = get_git_revision_hash(dir_path)
            all_commit_ids = get_entire_history(dir_path, curr_branch)

            for ix, commit_id in enumerate(all_commit_ids):
                result_ids.append(maybe_parse_diff.remote(
                    commit_id, org, out_file, project))

    total = sum(ray.get(result_ids))
    print('Total {}'.format(total))


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl Git repos")
    parser.add_argument("-i", "--in_dir", required=False, default="../data/Repos",
                        help="Directory to write to")
    parser.add_argument("-o", "--out_dir", required=False, default="../data/Diffs",
                        help="Repos to crawl through")
    parser.add_argument("-m", "--mode", required=False, default=CrawlMode.COMMENT_IN_DIFF,
                        help="Mode to crawl")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    mode.mode = args.mode
    ray.init(num_cpus=os.cpu_count() // 2)
    import time
    s = time.perf_counter()
    write_diffs(in_dir)
    parse(in_dir, out_dir)
    elapsed = time.perf_counter() - s
    print("{} executed in {:0.2f} seconds.".format(__file__, elapsed))
