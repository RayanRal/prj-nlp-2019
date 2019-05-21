const fs = require('fs');
const fetch = require('node-fetch');
const cheerio = require('cheerio');
const { queue } = require('async');
const { URL, resolve: urlResolve } = require('url');

const HREF_SELECTOR = '.pageContent [href^="threads"], .pageContent [href^="forums"]';
const POST_SELECTOR = '[id^="post-"]';

class Crawler {
  constructor(categoryId) {
    this.host = 'http://forum.lvivport.com/';
    this.id = categoryId;
    this.queued = new Set();
    this.queue = queue(this.getUrlContent.bind(this), 1);
    this.stream = fs.createWriteStream('data.txt', { flags: 'a' });

    this.queue.push(`${this.host}forums/${this.id}`);
    this.queue.drain = () => {
      this.stream.end();
      console.log(`${this.queued.size} urls visited`);
    };
  }

  processPost(el, url) {
    el.find('.messageContent').find('.bbCodeQuote, br, span').remove();
    const data = [];
    const user = el.find('.username.author[href]');
    if (user.length) {
      data.push(el.attr('id').split('-').pop());
      data.push(user && user.attr('href').split('.').pop().slice(0, -1));
      data.push(user && user.text());
      data.push(url);
      data.push(el.find('.messageText').text()
        .replace(/\n\s+/g, '\n').trim()
        .replace(/\n/g, '\\n'));
      this.stream.write(`${data.join('~!')}'\n'`);
    }
  }

  async getUrlContent(url) {
    const html = await fetch(url).then(res => res.text());
    const $ = cheerio.load(html);

    if (!$('body').hasClass(`node${this.id}`)) {
      return;
    }

    const hrefs = $(HREF_SELECTOR);
    const posts = $(POST_SELECTOR);

    hrefs.each((i, el) => {
      const href = urlResolve(this.host, el.attribs.href);
      const { origin, pathname } = new URL(href);
      const newUrl = origin + pathname;
      if (!this.queued.has(newUrl)) {
        this.queue.push(newUrl);
        this.queued.add(newUrl);
      }
    });

    posts.each((i, el) => {
      this.processPost($(el), url);
    });
  }
}

new Crawler(92);
