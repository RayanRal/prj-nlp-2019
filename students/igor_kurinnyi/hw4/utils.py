from itertools import islice


def jaccard_text_similarity(t1, t2, n):
    t1 = set(window(f'^{t1}$', n=n))
    t2 = set(window(f'^{t2}$', n=n))
    return len(t1 & t2) / len(t1 | t2)


def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield ''.join(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield ''.join(result)


if __name__ == '__main__':
    s = 'hello, freak bitches'
    d = 'hello, bitches'
    print(jaccard_text_similarity(s, d, 3))
