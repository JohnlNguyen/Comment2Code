

test-extract-code:
	@python3 Data/commit_files_crawler.py -t=True -db=True

get-diffs:
	@cd Data && python3 git_crawler.py

extract:
	@cd Data && python3 commit_files_crawler.py

inspect:
	@cd Data && python3 inspect_data.py
