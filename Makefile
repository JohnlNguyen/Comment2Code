
crawl:
	@cd Data && python3 git_crawler.py

clone:
	@cd Data && python3 cloner.py

get-projects:
	@cd Data && python3 clone_repos.py

extract:
	@cd Data && python3 code_crawler.py

inspect:
	@cd Data && python3 inspect_data.py

inspect-rand:
	@cd Data && python3 inspect_data.py --rand=True

test-extract:
	@cd Data && python3 code_crawler.py -d="../test/Diffs" -r="../test/Repos"

test-crawl:
	@cd Data && python3 git_crawler.py --out_dir="../test/Diffs" --in_dir="../test/Repos"  && \
	python3 ../test/git_crawler_test.py

