import subprocess
import os
import sys

projects = []
with open("projects.txt", "r") as f:
	projects = f.read().split("\n")

names = []
for project in projects:
	name = project.split("\t")[0]
	names.append(name)

for name in names:
	path = 'Repos/{}'.format(name)
	os.makedirs(path, exist_ok=True)
	url = 'https://github.com/{}'.format(name)
	subprocess.call(['git', 'clone', '-q', url, path], stdout=sys.stdout)
