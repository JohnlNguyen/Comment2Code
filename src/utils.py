from pygments.token import Comment, Text, Name, Keyword, Operator


def is_a_comment_line(ttypes):
    return not is_a_code_line(ttypes) and any([t in Comment for t in ttypes])


def contains_a_comment(ttypes):
    return any([t in Comment for t in ttypes])


def is_a_code_line(ttypes):
    return Name in ttypes or Operator in ttypes or Keyword in ttypes


def filter_comments(comments):
    # filter out to keep comments only one last time, edge case would be "#" in string
    return [c for c in comments if "#" in c]


def filter_code(code):
    # filter code to keep lines without #, edge case would same as filter_comment
    return [c for c in code if "#" not in c]


def is_a_comment_line_python(ttypes):
    return not is_a_code_line(ttypes) and any([t in Comment for t in ttypes])


def is_a_comment_line_java(ttypes):
    return not is_a_code_line(ttypes) and any([t in Comment for t in ttypes])


def contains_a_comment(ttypes):
    return any([t in Comment for t in ttypes])


def is_a_code_line(ttypes):
    return Operator in ttypes or Keyword in ttypes


def filter_comments(comments):
    # filter out to keep comments only one last time, edge case would be "#" in string
    return [c for c in comments if "#" in c]


def filter_code(code):
    # filter code to keep lines without #, edge case would same as filter_comment
    return [c for c in code if "#" not in c]


def get_tokens(lexer, line):
    return [t for t, _ in lexer.get_tokens(line)]


def not_docstring(line):
    return ">>>" not in line and ":" not in line
