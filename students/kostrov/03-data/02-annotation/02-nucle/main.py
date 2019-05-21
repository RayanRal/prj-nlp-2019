import itertools
from collections import defaultdict

f = open('./data/official-2014.combined-withalt.m2', 'r')

def hash_annotation(i, string):
    temp, err, edit, _, _, annotator = string.split('|||')
    _, start, end = temp.split(' ')
    return (annotator, err, '~'.join([str(i), start, end, err, edit]))

def process_annotations():
    data = defaultdict(set)
    data_extended = defaultdict(lambda: defaultdict(set))
    errors_list = set()
    sentence_number = 1
    for line in f:
        if line.startswith('A '):
            annotator, err, annotation_hash = hash_annotation(sentence_number, line.strip())
            errors_list.add(err)
            data[annotator].add(annotation_hash)
            data_extended[annotator][err].add(annotation_hash)
        elif line == '\n':
            sentence_number += 1
    if 'noop' in errors_list: errors_list.remove('noop')
    return errors_list, data, data_extended


def calculate_iaa_naive(set1, set2):
    intersection = len(set.intersection(set1, set2))
    union = len(set.union(set1, set2))
    return intersection / union

def calculate_iaa_kappa(set1, set2):
    len_set1, len_set2 = len(set1), len(set2)
    intersection = len(set.intersection(set1, set2))
    union = len(set.union(set1, set2))
    p_o = intersection / union
    p_y = (len_set1*len_set2)/(union*union)
    p_n = ((union - len_set1)*(union - len_set2))/(union*union)
    p_e = p_y + p_n
    return (p_o - p_e)/(1 - p_e) if p_e != 1 else 0

errors_list, data, data_extended = process_annotations()
combinations = list(itertools.combinations(data.keys(), 2))

iaa = [calculate_iaa_naive(data[ids[0]], data[ids[1]]) for ids in combinations]
print('naive IoU', sum(iaa)/len(iaa))

iaa_kappa = [calculate_iaa_kappa(data[ids[0]], data[ids[1]]) for ids in combinations]
print('using Cohen\'s Kappa', sum(iaa_kappa)/len(iaa_kappa))

result_extended = defaultdict(set)
for ids in combinations:
    temp = {}
    for error in errors_list:
        set1, set2 = data_extended[ids[0]][error], data_extended[ids[1]][error]
        if len(set1) and len(set2):
            result_extended[error].add(calculate_iaa_kappa(set1, set2))

iaa_kappa_by_error = {key: sum(value)/len(value) for key, value in result_extended.items()}
print('using Cohen\'s Kappa separated by error', iaa_kappa_by_error)
