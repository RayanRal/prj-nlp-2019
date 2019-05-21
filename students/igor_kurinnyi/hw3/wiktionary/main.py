import re
import json
import xml.sax
from collections import defaultdict


class MovieHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.wiktionary = defaultdict(list)
        self.title = ''

        self.current_tag = None
        self.is_italian = None
        self.is_sinonimi = False

    def startElement(self, tag, attributes):
        self.current_tag = tag

    def endElement(self, tag):
        if tag == 'page':
            self.title = ''
            self.is_italian = None
            self.is_sinonimi = False

        self.current_tag = None

    def characters(self, content):
        if self.current_tag == 'title':
            if ':' not in content:
                self.title = content
        elif self.current_tag == 'text':
            if self.title:
                self.check_language(content)
                self.check_sinonimi_start(content)
                self.extract_sinonimi(content)
                self.check_sinonimi_end(content)

    def check_language(self, content):
        if self.is_italian is None:
            self.is_italian = True if re.search(r'{{-it-}}', content) else False

    def check_sinonimi_start(self, content):
        if self.is_sinonimi and content.startswith('{{-'):
            self.is_sinonimi = False

    def check_sinonimi_end(self, content):
        if self.is_italian and (content == '{{-sin-}}'):
            self.is_sinonimi = True

    def extract_sinonimi(self, content):
        if self.is_sinonimi:
            matches = re.finditer(r'\[\[.+?\]\]', content)
            subsins = [m.group(0)[2:-2] for m in matches]
            if subsins:
                self.wiktionary[self.title].append(subsins)

    def endDocument(self):
        with open('sinonimi.json', 'w', encoding='utf-8') as f:
            json.dump(self.wiktionary, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    Handler = MovieHandler()
    parser.setContentHandler(Handler)

    parser.parse("../../data/wikimedia/itwiktionary-20190320-pages-articles-multistream.xml")
