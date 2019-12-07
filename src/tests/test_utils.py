import pygments

import sys
sys.path.append("..")

from lexer import build_lexer
from utils import *


def tests():
    lexer = build_lexer('python')
    line = 'print(a) # same line'
    ttypes = [t for t, _ in pygments.lex(line, lexer)]
    assert not is_a_comment_line(ttypes)
    assert contains_a_comment(ttypes)
    assert is_a_code_line(ttypes)

    line = '# comment line'
    ttypes = [t for t, _ in pygments.lex(line, lexer)]
    assert is_a_comment_line(ttypes)
    assert contains_a_comment(ttypes)
    assert not is_a_code_line(ttypes)

    line = 'x = 1'
    ttypes = [t for t, _ in pygments.lex(line, lexer)]
    assert not is_a_comment_line(ttypes)
    assert not contains_a_comment(ttypes)
    assert is_a_code_line(ttypes)

    line = '# ImportError: the machinery told us it does not exist'
    ttypes = [t for t, _ in pygments.lex(line, lexer)]
    assert is_a_comment_line(ttypes)
    assert contains_a_comment(ttypes)
    assert not is_a_code_line(ttypes)

    print('Test Passed')

if __name__ == '__main__':
    tests()
