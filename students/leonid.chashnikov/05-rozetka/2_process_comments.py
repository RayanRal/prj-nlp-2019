import pickle
import langdetect
from word_utils import _get_sentence_tone, _get_sentence_normal_forms


def _read_comments():
    result = []
    with open('./data/comments.p', 'rb') as f:
        while 1:
            try:
                result.append(pickle.load(f))
            except EOFError:
                break  # no more data in the file
    flat_result = [item for sublist in result for item in sublist]
    return flat_result


def _filter_out_by_lang(comments):
    results = []
    for comment in comments:
        try:
            lang = langdetect.detect(comment['text'])
            if lang == 'uk':
                results.append(comment)
        except Exception as e:
            print(e)
    return results


def _classify(ukr_comments):
    result = []
    for comment in ukr_comments:
        if comment.get('rating') in ('4', '5'):
            result.append(dict(label='pos', text=comment['text']))
        if comment.get('rating') == '3':
            result.append(dict(label='neu', text=comment['text']))
        if comment.get('rating') in ('1', '2'):
            result.append(dict(label='neg', text=comment['text']))
    return result


def _save_comments(classified_comments):
    with open('./data/processed_comments_usual.p', 'wb') as fp:
        pickle.dump(classified_comments, fp)


def _normalize_text(comments):
    for comment in comments:
        text_normalized = _get_sentence_normal_forms(comment['text'])
        tone = _get_sentence_tone(text_normalized)
        # comment['text'] = ' '.join(text_normalized)
        comment['features'] = text_normalized.append(tone)
        comment['tone'] = tone
    return comments


if __name__ == "__main__":
    comments = _read_comments()
    ukr_comments = _filter_out_by_lang(comments)
    classified_comments = _classify(ukr_comments)
    classified_comments = _normalize_text(classified_comments)
    _save_comments(classified_comments)
