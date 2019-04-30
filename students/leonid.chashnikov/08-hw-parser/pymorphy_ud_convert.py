DET = ['інакший', 'його', 'тамтой', 'чий', 'їх', 'інш.', 'деякий', 'ввесь', 'ваш',
       'ніякий', 'весь', 'інший', 'чийсь', 'жадний', 'другий', 'кожний',
       'такий', 'оцей', 'скілька', 'цей', 'жодний', 'все', 'кілька', 'увесь',
       'кожній', 'те', 'сей', 'ін.', 'отакий', 'котрий', 'усякий', 'самий',
       'наш', 'усілякий', 'будь-який', 'сам', 'свій', 'всілякий', 'всенький', 'її',
       'всякий', 'отой', 'небагато', 'який', 'їхній', 'той', 'якийсь', 'ин.', 'котрийсь',
       'твій', 'мій', 'це']

PREP = ["до", "на"]

mapping = {"ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ", "PRTF": "ADJ",
           "PRTS": "ADJ", "GRND": "VERB", "NUMR": "NUM", "ADVB": "ADV",
           "NPRO": "PRON", "PRED": "ADV", "PREP": "ADP", "PRCL": "PART"}

def normalize_pos(word):
    if word.tag.POS == "CONJ":
        if "coord" in word.tag:
            return "CCONJ"
        else:
            return "SCONJ"
    elif "PNCT" in word.tag:
        return "PUNCT"
    elif word.normal_form in PREP:
        return "PREP"
    else:
        return mapping.get(word.tag.POS, word.tag.POS)