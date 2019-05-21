from itertools import combinations
from dataclasses import field, astuple, dataclass

from error_type import ErrorType


@dataclass
class Annotation:
	span : tuple
	err_type : ErrorType
	correction : str
	annotator_id : int = field(compare=False)

	def __hash__(self):
		return hash(astuple(self))


class AnnotationSet(set):

	def __contains__(self, x):
		for item in self:
			if x == item:
				return True

	def __and__(self, other):
		intersect = [x for x in self if x in other]
		return AnnotationSet(intersect)

	def __str__(self):
		anns_str = ''.join([f'A: {a}\n' for a in self])
		return anns_str[:-1]

	@property
	def annotators_ids(self):
		return list(set(ann.annotator_id for ann in self))

	def filter_by(self, **params):
		result = list(self)
		for key, value in params.items():
			result = list(filter(lambda x: getattr(x, key) == value, result))
		return AnnotationSet(result)

	def cross_agreement(self):
		filters_key = [{'ann1' : {'annotator_id': id1}, 
		 				'ann2' : {'annotator_id': id2},
		 				'key'  : (id1, id2)} 
		 				for id1, id2 in self.annotator_pairs()]
		return self.all_agreements(filters_key)

	def cross_agreement_by_error(self):
		filters_key = [{'ann1': {'annotator_id': id1, 'err_type': err},
						'ann2': {'annotator_id': id2, 'err_type': err},
						'key': err}
					   for err in ErrorType
					   for id1, id2 in self.annotator_pairs()]
		return self.all_agreements(filters_key)

	def annotator_pairs(self):
		return combinations(sorted(self.annotators_ids), r=2)

	def all_agreements(self, filters_key):
		scores = dict()
		for fk in filters_key:
			ann1 = self.filter_by(**fk['ann1'])
			ann2 = self.filter_by(**fk['ann2'])
			if ann1 or ann2:
				scores[fk['key']] = ann1.agreement(ann2)
		return scores

	def agreement(self, other):
		return 2 * len(self & other) / (len(self) + len(other))


class Sentence:

	def __init__(self, s, anns):
		self.s = s
		self.anns = anns

	def correct(self, annotator_id):
		pass

	def __str__(self):
		return f'S: {self.s}\n{self.anns}'
