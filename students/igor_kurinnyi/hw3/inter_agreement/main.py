from collections import defaultdict, Counter
from corpus_reader import process_corpora

from pprint import pprint


NUCLE_PATH = '../data/conll14st-test-data/alt'
NUCLE_FILE = 'official-2014.combined-withalt.m2'
NUCLE = '{}/{}'.format(NUCLE_PATH, NUCLE_FILE)


def compute_corpora_agreement(corpora, atype='cross_agreement'):
	scores = defaultdict(Counter)
	for sent in corpora:
		sent_scores = getattr(sent.anns, atype)()
		for key, score in sent_scores.items():
			scores[key].update(score=score, count=1)
	return scores


def print_agreement(agreement):
	agreement = sorted(agreement.items(), key=lambda x: -x[1]['count'])
	for key, d in agreement:
		s = '{:>50} : {:.2f} ({})'
		print(s.format(str(key), d['score']/d['count'], d['count']))


if __name__ == '__main__':
	corpora = list(process_corpora(NUCLE))

	print('--- Inter-annotator agreement ---'.upper())
	scores = compute_corpora_agreement(corpora, 'cross_agreement')
	print_agreement(scores)

	print('\n--- Agreement for each error type ---'.upper())
	scores = compute_corpora_agreement(corpora, 'cross_agreement_by_error')
	print_agreement(scores)

	