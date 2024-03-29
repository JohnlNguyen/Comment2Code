
clean:
	@cd src && python3 clean.py

crawl:
	@cd src && python3 git_crawler.py

clone:
	@cd src && sh cloner.sh

get-projects:
	@cd src && python3 get_projects.py

extract:
	@cd src && python3 code_crawler.py

inspect:
	@cd src && python3 inspect_data.py -f='inspect_before_after.json'

inspect-rand:
	@cd src && python3 inspect_data.py --rand=True

test: test-crawl test-extract

test-extract:
	@cd src && python3 code_crawler.py -d="./Diffs" -r="./Repos"

test-crawl:
	@cd src && python3 git_crawler.py --out_dir="./Diffs" --in_dir="./Repos"  && \
	python3 test_git_crawler.py

