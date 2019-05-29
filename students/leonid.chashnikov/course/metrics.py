from random import randint
from statistics import mean

from textstat.textstat import textstatistics, easy_word_set, legacy_round, textstat
from os import listdir
from os.path import isfile, join

input_path = './raw_data/'
all_data_sents = []

files = [join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f))]
for filename in files:
    if filename.endswith('.txt'):
        with open(filename, encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
            text = text.replace('\n\n', '\n') \
                .replace('"', '') \
                .replace("\"", '') \
                .replace("...", "") \
                .replace("…", "") \
                .replace("—", "") \
                .replace(". ", " . ") \
                .replace(", ", " , ")
            # text = ' '.join(text.split("\n"))
            all_data_sents.append(text)

all_data_string = ' '.join(all_data_sents)
print(len(all_data_string))

splitted_data = all_data_string.split('\n')


flesh_score = []
gf_score = []

for x in range(10):
    index = randint(0, len(splitted_data))
    paragraph = all_data_string.split('\n')[index]
    flesh_score.append(textstat.flesch_reading_ease(paragraph))
    gf_score.append(textstat.gunning_fog(paragraph))

print('Mean Flesh {}'.format(mean(flesh_score)))  #50.06
print('Mean GF {}'.format(mean(gf_score)))  # 18.165


selected_par = "tyrion drank it in his window seat , brooding over the chaos of the kitchens below . the sun had not yet touched the top of the castle wall , but he could smell breads baking and meats roasting . the guests would soon be pouring into the throne room , full of anticipation; this would be an evening of song and splendor , designed not only to unite highgarden and casterly rock but to trumpet their power and wealth as a lesson to any who might still think to oppose joffrey's rule."

print('Flesh {}'.format(textstat.flesch_reading_ease(selected_par)))  # 0.09
print('GF {}'.format(textstat.gunning_fog(selected_par)))  # 36.18
# tyrion drank it in his window seat,
# brooding over the chaos of the kitchens below .
# the sun had not yet touched the top of the castle wall ,
# but he could smell breads baking and meats roasting .
# the guests would soon be pouring into the throne room ,
# full of anticipation; this would be an evening of song and splendor ,
# designed not only to unite highgarden and casterly rock but to
# trumpet their power and wealth as a lesson to any who might still think to oppose joffrey's rule.



baseline_par = "tyrion drank it in his window seat , where he sat drinking and watching the sea while the sun darkened over pyke . i have no place here , sam thought anxiously , when her red eyes fell upon him . someone had to help maester aemon up the steps . do not look at me , ever since that time i lost my horse . as if that could be helped . he was white and it was snowing , what did they expect the wind took that one , said grenn , another friend of lord snow is . try to hold the bow steady , sam. it is heavy , the fat boy complained , but he pulled the second arrow all the same . this one went high , sailing through the branches overhead , across the starry sky. snow, the moon murmured . the wolf made no answer . snow crunched beneath his feet . as  as you say , mlady. roose is not pleased . tell your bastard that. he is not my bastard , he wanted to say ."

print('Flesh {}'.format(textstat.flesch_reading_ease(baseline_par)))  #-69.95
print('GF {}'.format(textstat.gunning_fog(baseline_par)))  # 62.66
# input: tyrion drank it in his window seat ,
# output:
# tyrion drank it in his window seat, where he sat drinking and watching
# the sea while the sun darkened over pyke . i have no place here , sam thought anxiously ,
# when her red eyes fell upon him . someone had to help maester aemon up the steps .
# do not look at me , ever since that time i lost my horse .
# as if that could be helped . he was white and it was snowing ,
# what did they expect the wind took that one , said grenn ,
# another friend of lord snow is . try to hold the bow steady , sam.
# it is heavy , the fat boy complained , but he pulled the second arrow all the same .
# this one went high , sailing through the branches overhead , across the starry sky.
# snow, the moon murmured . the wolf made no answer . snow crunched beneath his feet .
# as  as you say , mlady. roose is not pleased . tell your bastard that. he is not my bastard , he wanted to say ."



charlevel_par = "tyrion drank it in his window seat, brooding over the sea . the storms and the first time it was the same and ser barristan said . i was the man who was a stone , and the sea would have been there , and the dragon had been a stream of a stone wall , and the storm remembered the starks and his face . the woman was the sea when the stag of a hundred leather and the sea with a current . they made the"

print('Flesh {}'.format(textstat.flesch_reading_ease(charlevel_par)))  #23.1
print('GF {}'.format(textstat.gunning_fog(charlevel_par)))  #33.39
# input: tyrion drank it in his window seat, brooding over
# output:
# "tyrion drank it in his window seat, brooding over the sea . the storms and the first time it was the same and
# ser barristan said . i was the man who was a stone , and the sea would have been there , and the dragon
# had been a stream of a stone wall , and the storm remembered the starks and his face . the woman was the
# sea when the stag of a hundred leather and the sea with a current . they made the"





