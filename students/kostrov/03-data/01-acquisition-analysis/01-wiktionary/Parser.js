const fs = require('fs');
const sax = require('sax');
const bz2 = require('unbzip2-stream');

const UA = './data/ukwiktionary-20190301-pages-articles-multistream.xml.bz2';
const ARABIC = './data/arwiktionary-20190301-pages-articles-multistream.xml.bz2';
const THAI = './data/thwiktionary-20190301-pages-articles-multistream.xml.bz2';

class Parser {
  constructor(filepath, {
    languageMarker = '{{=uk=}}',
    synonymsMarker = '==== Синоніми ====',
    sectionMarker = '==',
    dir = 'ltr',
  } = {}) {
    this.filepath = filepath;
    this.settings = {
      languageMarker,
      synonymsMarker,
      sectionMarker,
      dir,
    };

    this.currentParent = null;
    this.currentChild = null;
    this.currentChild = null;
    this.currentWord = null;
    this.result = {};

    this.saxStream = sax.createStream(true);
  }

  handleRtl(string) {
    return this.settings.dir === 'rtl' ? string.split('').reverse().join('') : string;
  }

  getMatches(string, regex, index = 1) {
    const matches = [];
    let match;
    while (match = regex.exec(string)) {
      matches.push(...this.handleRtl(match[index]).split('|'));
    }
    return matches;
  }

  extractSynonyms(text) {
    const synonyms = [];
    let synSectionFound = false;

    for (const line of text.split('\n')) {
      if (line === this.settings.synonymsMarker) {
        synSectionFound = true;
      } else if (synSectionFound) {
        if (line.startsWith(this.settings.sectionMarker)) {
          break;
        }
        const cleanLine = line
          .replace(/[{=('<](.*)[=})'>]/g, '')
          .replace(/[#*…?—\-.]/g, '')
          .replace(/\[http.+\]/g, '')
          .replace(/(\[\[:?\w.+]])/g, '')
          .replace(/:{2,}.*/g, '')
          .replace(/\s+/g, ' ')
          .trim();
        if (/\[\[/.test(cleanLine)) {
          synonyms.push(...this.getMatches(line, /\[\[+([^\]]+)\]/g));
        } else {          
          synonyms.push(...cleanLine.split(/[,;]/).map(item => 
            item
            .replace(/[\[\]]/g, '')
            .trim()
          ));
        }
      }
    }

    return synonyms.filter(item => item && item !== '-');
  }

  async extract() {
    fs.createReadStream(this.filepath)
      .pipe(bz2())
      .pipe(this.saxStream);

    this.saxStream.on('opentag', this.onOpenTag.bind(this));
    this.saxStream.on('closetag', this.onCloseTag.bind(this));
    this.saxStream.on('text', this.onText.bind(this));

    return new Promise((resolve, reject) => {
      this.saxStream.on('end', () => {
        resolve(this.result);
      });

      this.saxStream.on('error', (err) => {
        reject(err);
      });
    });
  }

  onOpenTag(node) {
    const { name } = node;
    switch (name) {
      case 'page':
        this.currentParent = 'page';
        break;
      case 'title':
        this.currentChild = 'title';
        break;
      case 'text':
        this.currentChild = 'text';
        break;
    }
  }

  onCloseTag(tag) {
    switch (tag) {
      case 'page':
        this.currentParent = null;
        this.currentChild = null;
        break;
      case 'title':
      case 'text':
        this.currentChild = null;
    }
  }

  onText(text) {
    switch (this.currentChild) {
      case 'title':
        this.currentWord = this.handleRtl(text);
        break;
      case 'text':
        if (text.startsWith(this.settings.languageMarker)) {
          const synonyms = this.extractSynonyms(text);
          if (synonyms.length) {
            this.result[this.currentWord] = synonyms;
          }
          this.currentWord = null;
        }
    }
  }
}

(async () => {
  fs.writeFileSync('./output/ua.json', JSON.stringify(await new Parser(UA).extract(), null, 2));

  fs.writeFileSync('./output/arabic.json', JSON.stringify(await new Parser(ARABIC, {
    languageMarker: '{{عربية}}',
    synonymsMarker: '=== مرادفات ===',
    dir: 'rtl',
  }).extract(), null, 2));

  fs.writeFileSync('./output/thai.json', JSON.stringify(await new Parser(THAI, {
    languageMarker: '== ภาษาไทย ==',
    synonymsMarker: '==== คำพ้องความหมาย ====',
  }).extract(), null, 2));
})();
