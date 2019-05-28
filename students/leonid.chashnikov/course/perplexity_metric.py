import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

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

# tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in all_data_string]
tokenized_string = all_data_string.split(' ')

n = 20
train_data = [nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in
              tokenized_string]
words = [word for sent in tokenized_string for word in sent]
words.extend(["<s>", "</s>"])
padded_vocab = Vocabulary(tokenized_string)
model = MLE(n)
model.fit(train_data, padded_vocab)

# test_sentences = ['an apple', 'an ant']
baseline_par = "tyrion drank it in his window seat , where he sat drinking and watching the sea while the sun darkened over pyke . i have no place here , sam thought anxiously , when her red eyes fell upon him . someone had to help maester aemon up the steps . do not look at me , ever since that time i lost my horse . as if that could be helped . he was white and it was snowing , what did they expect the wind took that one , said grenn , another friend of lord snow is . try to hold the bow steady , sam. it is heavy , the fat boy complained , but he pulled the second arrow all the same . this one went high , sailing through the branches overhead , across the starry sky. snow, the moon murmured . the wolf made no answer . snow crunched beneath his feet . as  as you say , mlady. roose is not pleased . tell your bastard that. he is not my bastard , he wanted to say ."

baseline_tokenized = baseline_par.split(' ')
# tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in baseline_par]

# test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
# for test in test_data:
#     print("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

test_data = [nltk.ngrams(t, n, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in
             baseline_tokenized]
for i, test in enumerate(test_data):
    print("PP({0}):{1}".format(baseline_par[i], model.perplexity(test)))
