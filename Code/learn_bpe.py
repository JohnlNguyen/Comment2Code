from bpe import Encoder
from util import get_data
from pdb import set_trace

import argparse
import json
import itertools


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("data", help="Path to data file")
	ap.add_argument("-v", "--vocabulary", help="Path to output vocab file")
	args = ap.parse_args()

	encoder = Encoder(vocab_size=32000, pct_bpe=1.0)

	with open(args.data) as f:
		data = json.load(f)

	data = list(get_data(data))
	data = list(itertools.chain.from_iterable(data))
	encoder.fit(data)
	encoder.save(args.vocabulary)


if __name__ == '__main__':
	main()
