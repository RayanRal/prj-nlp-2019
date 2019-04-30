import pronouncing

slang = {}


def get_celebrity_list(filename):
    celebrity_list = []
    with open(filename, "r") as input_file:
        for line in input_file:
            celebrity_list.append(line.replace('\n', ''))
    return celebrity_list


def generate_slang(celebrity_list):
    for name in celebrity_list:
        if name.split(' ')[-1].lower() in rhymes:
            if noun in slang.keys():
                slang[noun].append(name)
            else:
                slang[noun] = [name]


twitter_celebrities = get_celebrity_list('twitter_celebrity_dataset (post-processed).txt')
english_celebrities = get_celebrity_list('english_celebrities (post-processed).txt')

with open('top_nouns.txt', "r") as nouns:
    for noun in nouns:
        noun = noun.replace('\n', '')
        rhymes = pronouncing.rhymes(noun)
        generate_slang(twitter_celebrities)
        generate_slang(english_celebrities)

with open('chitty_chitty.txt', "w") as output_file:
    for key, value in slang.items():
        value = list(set(value))
        output_file.write('{}: {}\n'.format(key, value))

