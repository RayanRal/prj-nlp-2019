import pronouncing
import json

slang = {}


def get_dataset_list(filename):
    dataset_list = []
    with open(filename, "r") as input_file:
        for line in input_file:
            dataset_list.append(line.strip())
    return dataset_list


def generate_slang(dataset_list, dataset_label):
    for word in rhymes:
        for name in dataset_list:
            if name.split(' ')[-1].lower().endswith(word) and name.split(' ')[-1].lower() != noun.lower():
                if noun in slang.keys():
                    slang[noun].append((name, dataset_label))
                else:
                    slang[noun] = [(name, dataset_label)]


twitter_celebrities = get_dataset_list('twitter_celebrity_dataset (post-processed).txt')
english_celebrities = get_dataset_list('english_celebrities (post-processed).txt')
idioms = get_dataset_list('processed_idioms.txt')

with open('CRS_Dictionary.txt') as json_file:
    data = json.load(json_file)
    for noun in data.keys():
        rhymes = pronouncing.rhymes(noun)
        generate_slang(twitter_celebrities, 1)
        generate_slang(english_celebrities, 2)
        generate_slang(idioms, 3)

with open('chitty_chitty.txt', "w") as output_file:
    for key, value in slang.items():
        value = list(set(value))
        value = sorted(value, key=lambda value: value[1])
        output_file.write('{}: {}\n'.format(key, value))


accuracy = len(slang) / len(data)

# accuracy = 0.64


# Expanding dataset with idioms increased accuracy from 0.5 to 0.64
