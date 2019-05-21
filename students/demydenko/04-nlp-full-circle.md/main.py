import requests
from bs4 import BeautifulSoup
import pickle
import re
import en_core_web_md
# import en_core_web_lg
import spacy


def parse_data(boxer_wiki_name):
    website_url = requests.get("https://en.wikipedia.org/wiki/"+boxer_wiki_name).text
    soup = BeautifulSoup(website_url, "html.parser")

    # remove all <sup> tag
    for sup in soup("sup"):
        sup.decompose()

    page_main_content = soup.find('div', id='mw-content-text')

    # skip headers, leave only <p>
    page_paragraphs = page_main_content.find_all(['p'])

    # parse results table
    tables = soup.find_all('tbody')
    table_headers_patern = ["No.", "Result", "Record", "Opponent", "Type", "Round, time","Date", "Age", "Location", "Notes"]
    fight_result_list = []
    for table in tables:
        table_headers = [str(t.getText()).strip() for t in table.find_all("th")]
        if table_headers[:4] == table_headers_patern[:4]:
            # print(table_headers)
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                if len(cols) > 0:
                    fight_result_list.append((cols[0], cols[1], cols[3]))
            break

    defeated_true_list = [f[2] for f in fight_result_list if f[1] == 'Win']
    full_oponents_list = [f[2] for f in fight_result_list]
    # print(defeated_true_list)
    defeated_mentioned_list = set()
    text_paragraphs = []
    for page_paragraph in page_paragraphs:

        # remove tag <a> and count part of mentioned opponents
        for a in page_paragraph.find_all('a'):
            a_text = a.get_text() #  also remove list and tables.
            if a_text in defeated_true_list:
                defeated_mentioned_list.add(a_text)
            a.replace_with(a_text)

        text_paragraphs.append(page_paragraph.get_text())

    # pickle_out = open("dict.pickle", "wb")

    return (text_paragraphs, defeated_mentioned_list, defeated_true_list)



# text, defeated_mentioned_list, defeated_true_list = parse_data("Mike_Tyson")
# text, defeated_mentioned_list, defeated_true_list = parse_data("Evander_Holyfield")
text, defeated_mentioned_list, defeated_true_list = parse_data("Wladimir_Klitschko")

print('loading model...')
# nlp = en_core_web_md.load()
nlp = spacy.load('en_core_web_md')
print('model has been loaded')


merge_ents = nlp.create_pipe("merge_entities")
nlp.add_pipe(merge_ents)

defeated_list = set()
defeated_mentioned_list_2 = set()


docs = [nlp(t) for t in text]
for i, doc in enumerate(docs):
    print(doc.text)
    for j, sent in enumerate(doc.sents):
        # print('---------------------------------')
        # print(sent.text)

        ners = [ent.text for ent in sent.ents if ent.label_=='PERSON']
        for n in ners:
            # count rest of mentioned opponents
            if n in defeated_true_list:
                defeated_mentioned_list_2.add(n)


        subjects = [w for w in sent if w.dep_ == 'nsubj']
        for subject in subjects:
            print('document.sentence: {}.{}, subject: {}, action: {}'.format(i, j, subject.text, subject.head.text))

            # rule 1
            defeated_list_ = [str(c) for c in subject.head.children if str(c) in ners and c.dep_=='dobj' and subject.head.text=='defeated']
            [defeated_list.add(d) for d in defeated_list_]

            #rule 2
            defeated_list_ = [str(c) for c in subject.head.children if
                              str(c) in ners and c.dep_ == 'dobj' and subject.head.text == 'beating']
            [defeated_list.add(d) for d in defeated_list_]

            # rule_3
            defeated_list_ = [str(c) for c in subject.head.children if
                              str(c) in ners and c.dep_ == 'dobj' and subject.head.text == 'beat']
            [defeated_list.add(d) for d in defeated_list_]
            defeated_list_ = [str(c) for c in subject.head.children if
                              str(c) in ners and c.dep_ == 'dobj' and subject.head.text == 'stopping']
            [defeated_list.add(d) for d in defeated_list_]
            defeated_list_ = [str(c) for c in subject.head.children if
                              str(c) in ners and c.dep_ == 'dobj' and subject.head.text == 'knocked']
            [defeated_list.add(d) for d in defeated_list_]


defeated_mentioned_list_full = set.union(defeated_mentioned_list, defeated_mentioned_list_2)
true_positive = set.intersection(defeated_list, defeated_mentioned_list_full)

print("defeated_mentioned_list_full", defeated_mentioned_list_full)
print("------------------------------------------------")
print("defeated_list", defeated_list)
print("true_positive list", true_positive)
print("true_positive count", len(true_positive))
print("true_positive / len(defeated_mentioned_list_full)", len(true_positive)/len(defeated_mentioned_list_full))

pass

# RESULTS for Klichko

# defeated_list {'Leapai', 'Jennings', 'Wolfgramm', 'Sultan Ibragimov', 'DaVarryl Williamson', 'Eliseo Castillo',
# 'Lamon Brewster', 'Bryant Jennings', 'Molina', 'Paea Wolfgramm', 'Ray Austin', 'Chris Byrd'}

# true_positive list {'Sultan Ibragimov', 'Lamon Brewster', 'DaVarryl Williamson', 'Eliseo Castillo', 'Bryant Jennings', 'Paea Wolfgramm', 'Ray Austin', 'Chris Byrd'}
# true_positive count 8
# true_positive / len(defeated_mentioned_list_full) 0.32
