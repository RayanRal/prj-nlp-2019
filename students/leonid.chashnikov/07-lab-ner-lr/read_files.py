def get_locations(file):
    result = []
    with open(file, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
        for l in lines:
            words = l.split(',')
            for w in words:
                result.append(w)
    return result
