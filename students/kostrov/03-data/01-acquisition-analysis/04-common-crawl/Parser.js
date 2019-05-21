const fs = require('fs');
const subtag = require('subtag');
const cheerio = require('cheerio');
const { WARCGzParser } = require('node-warc');

class Parser {
  constructor(folder) {
    this.folder = folder;
    this.recordsCount = 0;
    this.nonHTMLCount = 0;
    this.stream = fs.createWriteStream('./output/data.csv', { flags: 'a' });
  }

  static getLanguage($) {
    const $html = $('html');

    return $html.attr('lang') || $html.attr('xml:lang')
      || $('meta[http-equiv="Content-Language"]').attr('content')
      || $('meta[property="og:locale"]').attr('content')
      || '';
  }

  processRecord(data) {
    this.recordsCount += 1;
    if (data.warcHeader['WARC-Identified-Payload-Type'] === 'text/html' && data.content) {
      const content = data.content.toString();
      const statusCode = +data.httpInfo.statusCode;
      const $ = cheerio.load(content);
      const lang = this.constructor.getLanguage($).toLowerCase();
      const { language, region } = subtag(lang);
      this.stream.write(`${statusCode},${language},${region}\n`);
    } else {
      this.nonHTMLCount += 1;
    }
  }

  async processFile(file) {
    return new Promise((resolve) => {
      const parser = new WARCGzParser(file);
      parser
        .on('record', (record) => {
          this.processRecord(record);
        })
        .on('done', () => {
          console.log('finished');
          resolve();
        })
        .on('error', (error) => {
          console.log(error);
        });
      parser.start();
    });
  }

  run() {
    fs.readdir(this.folder, async (err, files) => {
      for await (const file of files) {
        console.log(file);
        try {
          await this.processFile(`${this.folder}/${file}`);
        } catch (e) {
          console.log(e);
        }
      }
      this.stream.end();
      console.log(this.recordsCount);
      console.log(this.nonHTMLCount);
    });
  }
}

new Parser('./data').run();
