import spacy

nlp = spacy.load('en_core_web_md')

with open('processed_idioms.txt', "w") as output_file:
    with open('idioms.txt', "r") as input_file:
        for line in input_file:
            idiom = True
            doc = nlp(line)
            for token in doc:
                if token.pos_ == 'VERB':
                    idiom = False
                    break
            if idiom:
                output_file.write(line)


