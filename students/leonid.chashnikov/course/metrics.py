from textstat.textstat import textstatistics, easy_word_set, legacy_round, textstat


# readability
print(textstat.flesch_reading_ease(contents))
print(textstat.smog_index(contents), end="\n ")
print(textstat.gunning_fog(contents), end="\n ")

# perplexity - just accuracy? top 3, top 5?


