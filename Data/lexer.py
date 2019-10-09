import sys
import re
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, Text, String, Name
import fnmatch

def lex_file(file, language="python"):
	lexer = get_lexer_by_name(language)
	exts = lexer.filenames
	if language.lower() == 'plpgsql': exts.append('*.sql')
	cleaned_tokens = [[]]
	with open(file, "r", encoding='ascii', errors='ignore') as inFile:
		content = inFile.read()
		tokens = pygments.lex(content, lexer)
		inString = False
		for ttype, value in tokens:
			for _ in range(value.count("\n")): cleaned_tokens.append([])
			if ttype in Comment or ttype in String.Doc or ttype in Text: continue
			if ttype in String:
				if inString: continue
				value = '"str"'
				inString = True
			else: inString = False
			if re.match("\\s+", value): continue
			
			# Explicitly lex import/include paths if present
			if ttype in Name.Namespace:
				parts = value.split(".")
				if len(parts) > 1:
					for ix, p in enumerate(parts):
						if ix < len(parts) - 1:
							cleaned_tokens[-1].append(p)
							cleaned_tokens[-1].append(".")
						else:
							cleaned_tokens[-1].append(p.replace(";", ""))
							if ";" in p:
								cleaned_tokens[-1].append(";")
					continue
			
			# Do include pre-processor directives, but beware that they may need additional splitting on spaces
			if ttype in Comment.Preproc or ttype in Comment.PreprocFile:
				parts = value.split(" ")
				if len(parts) > 1:
					for p in parts:
						p = p.strip()
						if len(p) > 0: cleaned_tokens[-1].append(p)
					continue
			
			value = value.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r").strip()
			if len(value) > 0: cleaned_tokens[-1].append(value)
	return cleaned_tokens
