

test-extract-code:
	@python3 Data/commit_files_crawler.py -t=True -db=True

inspect:
	@cd Data && python3 inspect_data.py
