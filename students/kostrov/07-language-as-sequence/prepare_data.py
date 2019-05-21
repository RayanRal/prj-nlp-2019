import spacy
from spacy.matcher import Matcher

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_lg', disable=['ner', 'textcat'])

matcher = Matcher(nlp.vocab)
matcher.add('APOSTROPHED', None, [{'ORTH': "'"}, {'IS_ALPHA': True}])

def apostrophe_merger(doc):
    matched_spans = []
    matches = matcher(doc)
    for _, start, end in matches:
        span = doc[start:end]
        matched_spans.append(span)
    for span in matched_spans:
        span.merge()
    return doc

nlp.add_pipe(apostrophe_merger, first=True)

def prepare_data(sentences):
    doc = nlp(sentences)
    prepared_features = []
    labels = []
    left_punct_opened = False
    doc_len = len(doc)
    idx = 0
    for token in doc:
        if token.is_left_punct:
            left_punct_opened = True
        elif token.is_right_punct:
            left_punct_opened = False

        if token.i != doc_len - 1:
            features = {}
            is_start = token.is_sent_start

            if is_start:
                idx = 0
                left_punct_opened = False

            if token.i == 0:
                features['-1'] = 'BGN'
                features['-1.istitle'] = False
            elif is_start:
                prev = token.nbor(-2)
                features['-1'] = prev.text
                labels[-1] = True
                features['-1.istitle'] = prev.is_title
            else:
                prev = token.nbor(-1)
                features['-1'] = prev.text
                features['-1.istitle'] = prev.is_title

            if token.i == doc_len - 2:
                features['+1'] = 'ND'
                features['+1.istitle'] = True
            else:
                nxt = token.nbor(1)
                features['+1'] = nxt.text
                features['+1.istitle'] = nxt.is_title

            head = token.head
            features['head_text'] = head.text
            features['head_pos'] = head.pos
            features['head_tag'] = head.tag
            features['text'] = token.text
            features['pos'] = token.pos
            features['tag'] = token.tag
            features['dep'] = token.dep
            features['idx'] = idx
            features['is_upper'] = token.is_upper
            features['is_title'] = token.is_title
            features['left_punct_opened'] = left_punct_opened
            features['has_children'] = len(list(token.children)) > 0
            
            labels.append(False)
            prepared_features.append(features)
            idx += 1
    return (labels, prepared_features)
