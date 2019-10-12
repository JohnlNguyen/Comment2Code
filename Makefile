
crawl:
	@cd Data && python3 git_crawler.py

extract:
	@cd Data && python3 commit_files_crawler.py

inspect:
	@cd Data && python3 inspect_data.py

inspect-rand:
	@cd Data && python3 inspect_data.py --rand=True

test-extract-code:
	@python3 Data/commit_files_crawler.py -t=True -db=True

test-crawl:
	@cd Data && python3 git_crawler.py --out_dir="../test/Diffs" --in_dir="../test/Repos"

