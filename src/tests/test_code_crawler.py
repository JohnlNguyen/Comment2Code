from code_crawler import extract_code, capture_comment
from pathlib import Path
from lexer import build_lexer

import linecache


def tests():
	path = Path('./test_file.py')
	lexer = build_lexer()
	comment, code, h = extract_code(1, path)
	print("-" * 10)
	print(comment)
	print(code)
	assert comment == "# test case 1, starting at line 2\\n"
	assert code == [['def', 'hello', '(', ')', ':'], []]

	comment, code, h = extract_code(19, path)
	print("-" * 10)
	assert comment == "# ImportError: the machinery told us it does not exist\\n# ValueError:\\n#    - the module name was invalid\\n#    - the module name is __main__\\n#    - *we* raised `ValueError` due to `spec` being `None`\\n"
	assert code == [['except', '(', 'ImportError', ',', 'ValueError', ')', ':'], []]
	print(comment)
	print(code)

	comment, code, h = extract_code(40, path)
	print("-" * 10)
	assert comment == "# we were unable to find the `package_path` using PEP 451 loaders\\n"
	assert code == [['loader', '=', 'pkgutil', '.', 'get_loader', '(', 'root_mod_name', ')'],
					['if', 'loader', 'is', 'None', 'or', 'import_name', '==', '"str"', ':'],
					['if', 'loader', 'is', 'None', 'or', 'root_mod_name', '==', '"str"', ':'], []]
	print(comment)
	print(code)

	comment, code, h = extract_code(52, path)
	print("-" * 10)
	print(comment)
	print(code)
	assert comment == "# Google App Engine's HardenedModulesHook\\n#\\n# Fall back to imports.\\n"
	assert code == [['if', 'x', '==', 'True', ':'], ['x', '=', '1'], []]

	comment, code, h = extract_code(61, path)
	print("-" * 10)
	print(comment)
	print(code)

	comment, code, h = extract_code(64, path)
	print("-" * 10)
	print(comment)
	print(code)

	print("-" * 10)
	content = linecache.getlines(path.as_posix())
	comment, comment_end = capture_comment(content, lexer, 0)
	assert comment == ['# test case 1, starting at line 2\n']


if __name__ == '__main__':
	tests()
