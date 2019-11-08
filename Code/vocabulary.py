import util
import re
import argparse
import yaml
import json
import itertools
from bpe import Encoder
from pdb import set_trace

import csv


def get_tokens(data_path):
	""" quora """
	# with open(data_path, newline='') as csv_file:
	#     reader = csv.reader(csv_file, delimiter=',')
	#     next(reader)
	#     for line in reader:
	#         yield line[3]
	#         yield line[4]
	"""normal"""
	with open(data_path, 'r') as f:
		data = json.load(f)
		yield from util.get_data(data)


def main():
	# Extract arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--data", required=True, help="Path to data file")
	ap.add_argument("-v", "--vocabulary", required=False, help="Path to output vocab file")
	args = ap.parse_args()
	config = yaml.safe_load(open("config.yml"))

	vocab = VocabularyBuilder(config["vocabulary"], file_contents=list(
		get_tokens(args.data)), out_vocab_path=args.vocabulary)


class VocabularyBuilder(object):
	special_tokens = []

	def __init__(self, vocab_config, file_contents=None, vocab_path=None, out_vocab_path='vocab'):
		self.vocab_cutoff = vocab_config["vocab_cutoff"]
		self.split_kind = vocab_config["split_tokens"].lower()
		self.out_vocab_path = out_vocab_path

		if self.split_kind == "bpe":
			self.split_file = vocab_config["split_file"]
			self.bpe_limit = vocab_config["bpe_limit"]
		else:
			self.split_file = False
		if vocab_path != None:
			print("Reading vocabulary from base:", vocab_path)
			self.load_vocab(vocab_path)
		else:
			print("No vocabulary provided; building from training data")
			self.build_vocab(file_contents)

		self.vocab_dim = len(self.w2i)
		print("Vocabulary dimension:", self.vocab_dim)

		if self.split_kind == "bpe":
			self.bpe_lookup_dict = {}
			for token in self.w2i.keys():
				if not token:
					continue
				if token[0] not in self.bpe_lookup_dict:
					self.bpe_lookup_dict[token[0]] = set([token])
				else:
					self.bpe_lookup_dict[token[0]].add(token)

		self.num_oov = 0

	def vocab_key(self, w):
		if w in self.w2i:
			return self.w2i[w]
		else:
			print("Num oov ", self.num_oov)
			self.num_oov += 1
			return self.w2i["<unk>"]

	def tokenize(self, line):
		tokens = [line] if self.split_file else re.split('\s+', line)
		if self.split_kind == "bpe":
			subtokens = []
			for token in tokens:
				if token in self.special_tokens:
					subtokens.append(token)
					continue
				if not self.split_file:
					token += "#"
				ix = 0
				while ix < len(token):
					c = token[ix]
					if ord(c) < 33 or ord(c) >= 127:
						ix += 1
						continue
					candidates = self.bpe_lookup_dict[c]
					# Only sub-tokens that match the next characters and don't leave the
					# end-of-word marker left by itself
					candidates = [t for t in candidates if
								  t == token[ix:ix + len(t)] and (self.split_file or len(token) != ix + len(t) + 1)]
					top_candidate = max([(t, len(t)) for t in candidates],
										key=lambda e: e[1])[0] if candidates else []
					subtokens.append(top_candidate)
					ix += len(top_candidate)
			return subtokens
		elif self.split_kind == "heuristic":
			return [t for token in tokens for t in util.split_subtokens(token)]
		elif self.split_kind == "chars":
			return [t for token in tokens for t in token if ord(t) >= 33 and ord(t) < 127]
		else:
			return tokens

	def build_vocab(self, file_contents, write=True):
		if self.split_kind == "bpe" and self.split_file:
			self.token_counts = self.build_bpe_full_file(file_contents)
			top_words = sorted(self.token_counts.items(), key=lambda i: i[1], reverse=True)
			top_words = [t[0] for t in top_words]  # if t[1] >= self.vocab_cutoff]
		elif self.split_kind == "chars":
			top_words = [chr(c) for c in range(33, 127)]
			self.token_counts = {c: 1000 for c in top_words}
		else:
			self.token_counts = {}
			self.w2i = None
			for file in file_contents:
				for label in re.split('\s+', file):
					subtokens = [
						label] if self.split_kind != "heuristic" else util.split_subtokens(label)
					for sub in subtokens:
						util.merge(self.token_counts, sub, 1)

			# Ensure some key tokens make it into the vocabulary
			if "<unk>" not in self.token_counts:
				self.token_counts["<unk>"] = max(self.vocab_cutoff, sum(
					[c for c in self.token_counts.values() if c < self.vocab_cutoff]))
			for ix, s in enumerate(self.special_tokens):
				if s not in self.token_counts or self.split_kind == "bpe":
					self.token_counts[s] = int(1e9) + ix

			if self.split_kind == "bpe":
				print("Computing BPE on", len(self.token_counts), "tokens")
				self.token_counts = self.build_bpe(self.token_counts)
				top_words = sorted(self.token_counts.items(), key=lambda i: i[1], reverse=True)
				top_words = [t[0] for t in top_words]  # if t[1] >= self.vocab_cutoff]
			else:
				# Sort and discard tokens to infrequent to keep
				top_words = sorted(self.token_counts.items(), key=lambda t: t[1], reverse=True)
				top_words = [t[0] for t in top_words if t[1] >= self.vocab_cutoff]

		# Build the vocabulary
		self.w2i = {w: i for i, w in enumerate(top_words)}
		self.i2w = {i: w for w, i in self.w2i.items()}

		if write:
			self.save_vocab(self.out_vocab_path)

	def save_vocab(self, path):
		with open(path, "w", encoding="utf8") as f:
			for ix in range(len(self.w2i)):
				w = self.i2w[ix]
				f.write(str(self.token_counts[w]))
				f.write("\t")
				f.write(w)
				if ix < len(self.w2i) - 1:
					f.write('\n')

	def load_vocab(self, vocab_path):
		with open(vocab_path, "r", encoding="utf8") as f:
			vocab = [l.rstrip('\n').split("\t", 1) for l in f.readlines()]
			vocab = [l[1] for l in vocab if
					 l[0].isdigit() and int(l[0]) >= (0 if self.split_kind == "bpe" else self.vocab_cutoff)]

		self.w2i = {w: i for i, w in enumerate(vocab)}
		self.i2w = {i: w for w, i in self.w2i.items()}
		if not "<unk>" in self.w2i:
			self.w2i["<unk>"] = len(self.w2i)
			self.i2w[self.w2i["<unk>"]] = "<unk>"

	def build_bpe(self, token_counts):
		token_counts = [(list(t), c) for t, c in sorted(token_counts.items(),
														key=lambda i: i[1], reverse=True) if len(t) > 0]
		for ix in range(len(token_counts)):
			token_counts[ix][0][-1] += "#"
		bpe_pairs = {}
		count_table = {}
		loc_table = {}
		for tix, (token, count) in enumerate(token_counts):
			for ix, c in enumerate(token):
				if count >= 1e9:
					bpe_pairs["".join(token)[:-1]] = count
					continue
				util.merge(bpe_pairs, c, count)
				if ix > 0:
					pair = token[ix - 1] + c
					util.merge(count_table, pair, count)
					util.merge(loc_table, pair, tix, wrap_fn=lambda v: set([v]))
		for ix in range(33, 127):
			if chr(ix) not in bpe_pairs:
				bpe_pairs[chr(ix)] = 1
			if chr(ix) + "#" not in bpe_pairs:
				bpe_pairs[chr(ix) + "#"] = 1

		for step in range(self.bpe_limit):
			tc = 0
			top_pair = top_count = None
			for t, c in count_table.items():
				if t not in bpe_pairs and c > tc:
					top_pair, top_count = t, c
					tc = top_count
			if top_pair is None:
				break  # Typically means vocabulary is too small for this BPE cut-off
			bpe_pairs[top_pair] = top_count
			for tix in loc_table[top_pair]:
				token, token_count = token_counts[tix]
				ix = 1
				while ix < len(token):
					if token[ix - 1] + token[ix] == top_pair:
						if ix > 1:  # Update counts of preceding token, if any
							count_table[token[ix - 2] + token[ix - 1]] -= token_count
							util.merge(count_table, token[ix - 2] + top_pair, token_count)
							util.merge(loc_table, token[ix - 2] +
									   top_pair, tix, wrap_fn=lambda v: set([v]))
						if ix < len(token) - 1:
							count_table[token[ix] + token[ix + 1]] -= token_count
							util.merge(count_table, top_pair + token[ix + 1], token_count)
							util.merge(loc_table, top_pair +
									   token[ix + 1], tix, wrap_fn=lambda v: set([v]))
						# Finally, collapse the token and delete the remnant (so don't update ix)
						token[ix - 1] = top_pair
						del token[ix]
					else:
						ix += 1
		return bpe_pairs

	def build_bpe_full_file(self, strings):
		strings = [list(s) if isinstance(s, str) else s for s in strings]
		bpe_pairs = {}
		count_table = {}
		loc_table = {}
		for tix, token in enumerate(strings):
			if not token:
				continue
			for ix, c in enumerate(token):
				if c == '\n':
					c = '\\n'
				if c == '\t':
					c = '\\t'
				if c not in bpe_pairs:
					bpe_pairs[c] = 0
				bpe_pairs[c] += 1
				if ix == 0:
					continue
				pair = token[ix - 1] + c
				if pair not in count_table:
					count_table[pair] = 0
					loc_table[pair] = set()
				count_table[pair] += 1
				loc_table[pair].add((tix, ix))
		for ix in range(33, 127):
			if chr(ix) not in bpe_pairs:
				bpe_pairs[chr(ix)] = 1

		def get_context(string, cix, fwd=True):
			dir = 1 if fwd else -1
			for ix in range(cix + dir, len(string) if fwd else 0, dir):
				if string[ix] != '###':
					return ix
			return None

		for step in range(self.bpe_limit):
			tc = 0
			top_pair = top_count = None
			for t, c in count_table.items():
				if t not in bpe_pairs and c > tc:
					top_pair, top_count = t, c
					tc = top_count
			if top_pair is None:
				break  # Typically means vocabulary is too small for this BPE cut-off
			print("Adding\t'{0}'\t({1})".format(top_pair, top_count))
			bpe_pairs[top_pair] = top_count
			t_prev = ''
			new_tokens = set()  # For effective pruning
			for tix, cix in loc_table[top_pair]:
				string = strings[tix]
				prev_ix = get_context(string, cix, fwd=False)
				if prev_ix == None or string[prev_ix] + string[cix] != top_pair:
					continue  # May happen if cix-1 was fused by previous step
				# Update counts forward with knowledge of new pair
				prev_2_ix = get_context(string, prev_ix, fwd=False)
				if prev_2_ix != None:  # Update counts of preceding token, if any
					if string[prev_2_ix] + string[prev_ix] in count_table:
						count_table[string[prev_2_ix] + string[prev_ix]] -= 1
					new_token = string[prev_2_ix] + top_pair
					if new_token not in count_table:
						count_table[new_token] = 0
						loc_table[new_token] = set()
					count_table[new_token] += 1
					loc_table[new_token].add((tix, cix))
					new_tokens.add(new_token)
				# Update counts backward with knowledge of new pair
				next_ix = get_context(string, cix, fwd=True)
				if next_ix != None:
					if string[cix] + string[next_ix] in count_table:
						count_table[string[cix] + string[next_ix]] -= 1
					new_token = top_pair + string[next_ix]
					if new_token not in count_table:
						count_table[new_token] = 0
						loc_table[new_token] = set()
					count_table[new_token] += 1
					loc_table[new_token].add((tix, cix))
					new_tokens.add(new_token)
				# Finally, collapse the tokens and set the previous one to a 'blank' substitute
				string[cix] = top_pair
				string[prev_ix] = '###'
			for new_token in new_tokens:
				if count_table[new_token] < 10:  # TODO: don't hard-code count cutoff
					del count_table[new_token]
					del loc_table[new_token]
		return bpe_pairs


class BPE(object):
	def __init__(self, vocab_config, file_contents=None, vocab_path=None, out_vocab_path='vocab'):
		if vocab_path:
			self.encoder = self.load_vocab(vocab_path)
		else:
			self.encoder = Encoder(vocab_size=32000, pct_bpe=1.0, silent=False)

	def load_vocab(self, vocab_path):
		return Encoder.load(vocab_path)

	def save_vocab(self, path):
		self.encoder.save(path)

	def tokenize(self, line):
		return self.encoder.tokenize(line)

	def vocab_key(self, w):
		UNK = self.encoder.word_vocab[self.encoder.UNK]
		return self.encoder.bpe_vocab.get(w, UNK)

	def transform(self, line):
		return list(itertools.chain.from_iterable(self.encoder.transform(line, reverse=False, fixed_length=None)))

	@property
	def vocab_dim(self):
		return len(self.encoder.bpe_vocab)


if __name__ == '__main__':
	main()
