#!/usr/bin/python3

import xml.sax
import sys
import re

class SynonymsHandler(xml.sax.ContentHandler):
    """A SAX handler that parses words and their synonyms from German Wiktionary backup"""

    WORD_REGEXP = r'==\s+([^\(\[\]]+)\(\{\{Sprache\|Deutsch\}\}\)\s+=='
    SYN_REGEXP = r'\[\[([^\]]+)\]\]'

    @staticmethod
    def extract_synonyms(text):

        lines = text.split('\n')

        word = ""
        synonyms = []
        synonyms_start = False
        synonyms_stop = False

        for line in lines:
            if not word and not synonyms_start:
                word_match = re.search(SynonymsHandler.WORD_REGEXP, line)
                if (word_match):
                    word = word_match.group(1).strip()
            
            if not synonyms_start:
                if line.startswith('{{Synonyme}}'):
                    synonyms_start = True
            elif not synonyms_stop:
                if line.startswith(':'):
                    synonyms.extend(list(map(lambda w: w.strip(), re.findall(SynonymsHandler.SYN_REGEXP, line))))
                else:
                    synonyms_stop = True

        if word and synonyms:
            return (word, synonyms)     
        
        return None

    def __init__(self, ouput_file):
        self.ouput_file = ouput_file
        self.in_page = False
        self.in_text = False
        self.content = ""

    def startElement(self, tag, attributes):
        if tag =='page':
            self.in_page = True
        elif tag == 'text':
            self.in_text = True    

    def endElement(self, tag):
        if tag == 'page':
            self.in_page = False
        elif tag == 'text': 
            self.in_text = False
            word_synonyms = SynonymsHandler.extract_synonyms(self.content)
            self.content = ""
            if word_synonyms:
                self.save(word_synonyms)


    def save(self, word_synonyms):
        print(word_synonyms, file = self.ouput_file)      
    
    def characters(self, content):
        if self.in_page and self.in_text:
            self.content += content
            
    
if ( __name__ == "__main__"):
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    if len(sys.argv) < 2:
        print("Usage {} <wiktionary dump file> [output file]".format(sys.argv[0]))
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) > 2:
        ouput_file = sys.argv[2]
    else:
        ouput_file = 'synonyms.txt'

    with open(ouput_file, 'w') as out:
        parser.setContentHandler(SynonymsHandler(out))
        parser.parse(input_file)