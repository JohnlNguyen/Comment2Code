
train:
	@cd Code && python3 code_comment_aligner.py ../data/Pairs/code_comment_90742.json -v vocab

train-dcs:
	@cd Code && python3 code_comment_aligner.py ../data/python/final/jsonl/train/python_train_9.jsonl -v dcs_vocab

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
	@cd src && python3 inspect_data.py

inspect-100k:
	@cd src && python3 inspect_data.py -f='comment_code_100k.pkl'

merge:
	@cd src && python3 merge.py

inspect-rand:
	@cd src && python3 inspect_data.py --rand=True

test: test-crawl test-extract

test-extract:
	@cd src && python3 code_crawler.py -d="./Diffs" -r="./Repos"

test-crawl:
	@cd src && python3 git_crawler.py --out_dir="./Diffs" --in_dir="./Repos"  && \
	python3 test_git_crawler.py

