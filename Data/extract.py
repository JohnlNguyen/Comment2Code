import io
import tokenize


def extract(code):
	res = []
	comment = None
	stringio = io.StringIO(code)
	# pass in stringio.readline to generate_tokens
	for toktype, tokval, begin, end, line in tokenize.generate_tokens(stringio.readline):
		if toktype != tokenize.COMMENT:
			res.append((toktype, tokval))
		else:
			# wrap (toktype, tokval) tupple in list
			print(tokenize.untokenize([(toktype, tokval)]))
	return res
