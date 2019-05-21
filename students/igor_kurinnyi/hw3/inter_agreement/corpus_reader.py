from error_type import ErrorType
from sentence_annotation import Annotation, AnnotationSet, Sentence


def line_reader(file_name):
	with open(file_name, 'r') as f:
		for line in f:
			yield line.strip()


def process_corpora(data_file):
	data = line_reader(data_file)

	for row in data:
		if row.startswith('S'):
			s = row[2:]
			annotation_set = AnnotationSet()
		elif row.startswith('A'):
			ann = make_annotation(row[2: ])
			annotation_set.add(ann)
		else:
			yield Sentence(s=s, anns=annotation_set)


def make_annotation(s):
	span, err_type, correction, _, _, ann_id = s.split('|||')
	span = tuple(map(int, span.split(' ')))
	err_type = ErrorType[err_type.replace('-', '')]
	ann_id = int(ann_id)
	return Annotation(span, err_type, correction, ann_id)
