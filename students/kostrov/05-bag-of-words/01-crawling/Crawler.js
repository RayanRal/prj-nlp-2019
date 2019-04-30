const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const { queue } = require('async');
const fetch = require('node-fetch');
const querystring = require('querystring');

const CATEGORIES_URL = 'https://rozetka.com.ua/all-categories-goods/';

const cleanText = (string) => {
  return string
    .replace(/[\n\s\t]+/g, ' ')
    .replace(/^\-/, '')
    .trim();
}

class Crawler {
  constructor(url) {
    this.url = url;
    this.queued = new Set();
    this.queue = queue(this.getUrlContent.bind(this), 1);
    this.queue.drain = () => {
      this.stream.end();
      console.log(`${this.queued.size} urls visited`);
    };

    this.stream = fs.createWriteStream(path.resolve(__dirname, 'data', 'output', 'data2.tsv'), { flags: 'a' });
  }

  /**
   * Retrieving all categories
   * @param {String} url 
   */
  async getCategoriesList(url) {
    const categories = {};

    const html = await fetch(url).then(res => res.text());
    const $ = cheerio.load(html);
    $('.all-cat-b').each((i, section) => {
      const $section = $(section);
      const $sectionHref = $section.find('.all-cat-b-title-link').attr('href');
      categories[$sectionHref] = [];

      $section.find('.all-cat-b-l').each((j, column) => {
        let parent;
        $(column).find('[name="all-cat-i"]').each((k, category) => {
          const $category = $(category);
          const href = $category.attr('href');
          if ($category.hasClass('all-cat-b-l-i-link-parent')) {
            parent = href;
            categories[parent] = categories[parent] || [];
            if ($sectionHref !== href) {
              categories[$sectionHref].push(href);
            }
          } else if (href !== parent) {
            categories[parent].push(href);
          }
        });
      });
    });
    return categories;
  }

  /**
   * Getting all related categories for particular category
   * so we don't crawl irrelevant items
   * @param {String} url 
   * @returns {Array}
   */
  getRelevantCategories(url) {
    const result = [];
    const categories = this.categories[url];
    
    if (categories) {
      for (const item of categories) {
        if (!result.includes(item)) {
          result.push(...this.getRelevantCategories(item));
        }
      }
    }
    result.push(url);
    return result;  
  }

  constructCommentScrollUrl(url, number) {
    const scrollQs = {
      page: number,
      scroll: true,
      tab: 'comments',
      sort: 'date',
    };

    return `${url}${querystring.stringify(scrollQs, ';')}/`;
  }


  constructItemScrollUrl(url, number) {
    const scrollQs = {};

    if (number !== 1) {
      scrollQs.page = number;
    }      
    scrollQs.scroll = true;

    if (/=/.test(url)) {
      const urlParts = url.split('/');
      urlParts[urlParts.length - 2] = querystring.stringify(Object.assign({},
        scrollQs, 
        querystring.parse(urlParts[urlParts.length - 2], ';')
      ), ';');
      return urlParts.join('/');
    }

    return `${url}${querystring.stringify(scrollQs, ';')}/`;
  }

  getItems(html) {

  }

  queueCategories($, categories) {
    console.log('queueCategories');
    categories.each((i, el) => {
      const href = $(el).attr('href');
      if (!this.queued.has(href) && this.relevantCategories.has(href)) {
        this.queue.push(href);
        this.queued.add(href);
      }
    });
  }

  async getComments(url, commentsCount) {
    console.log('getComments', url);
    let pagesCount = Math.ceil(commentsCount / 10);
    while (pagesCount) {
      const comments = [];
      const html = await fetch(this.constructCommentScrollUrl(url, pagesCount))
        .then(res => res.text());
      const $ = cheerio.load(html);
      $('.pp-review-i').each((i, comment) => {
        const $comment = $(comment);
        const id = $comment.attr('name');
        const bought = $comment.find('.pp-review-buyer-note').length;
        const rating = $comment.find('meta[itemprop="ratingValue"]');
        const review = {
          text: null,
          adv: null,
          dis: null,
        };
        const upvotes = cleanText($comment.find('.pp-review-vote-positive .pp-review-vote-count').text()) || 0;
        const downvotes = cleanText($comment.find('.pp-review-vote-negative .pp-review-vote-count').text()) || 0;
        $comment.find('.pp-review-text-i').each((j, el) => {
          const $el = $(el);
          const span = $el.find('span.bold');
          if (span.length) {
            if (span.text() == 'Достоинства:') {
              span.remove();
              review.adv = cleanText($el.text());
            } else if (span.text() == 'Недостатки:') {
              span.remove();
              review.dis = cleanText($el.text());
            }
          } else {
            review.text = cleanText($el.text());
          }
        });
        if (rating.length && review.text) {
          comments.push([
            id,
            rating.attr('content'),
            bought,
            ...Object.values(review),
            upvotes,
            downvotes,
          ].join('\t'));
        }
      })
      pagesCount--;
      this.stream.write(`${comments.join('\n')}\n`);
    }
  }

  async getCommentsLinks(html) {
    console.log('getCommentsLinks');
    const $ = cheerio.load(html);
    for (const item of Array.from($('.g-rating-reviews-link'))) {
      const $item = $(item);
      const text = $item.text().trim()
      if ( text !== 'Оставить отзыв') {
        await this.getComments($item.attr('href'), +text.match(/\d+/));
      }
    }
  }

  async getGoods(url, i = 1) {
    console.log('getGoods', url);
    let scrollUrl = this.constructItemScrollUrl(url, i);
    let res = await fetch(scrollUrl);
    while (res.status == 200 && res.url === scrollUrl) {
      const html = await res.text();
      scrollUrl = this.constructItemScrollUrl(url, ++i);
      await this.getCommentsLinks(html);
      res = await fetch(scrollUrl);
    }
  }


  async crawlCategory(url) {
    const html = await fetch(url).then(res => res.text());
    const $ = cheerio.load(html);
    const subcategories = $('.pab-items-i-link');

    if (subcategories.length) {
      this.queueCategories($, subcategories);
    } else {
      await this.getGoods(url);
    }
  }

  async getUrlContent(url) {
    console.log(url);
    const html = await fetch(url).then(res => res.text());
    const $ = cheerio.load(html);
    const $itemContainer = $('.container');
    
    if ($itemContainer.length) {
      await this.crawlCategory(url);
    }
  }


  async start() {
    this.categories = await this.getCategoriesList(CATEGORIES_URL);
    this.relevantCategories = new Set(this.getRelevantCategories(this.url));
    this.queue.push(this.url);
  }  
}

new Crawler('https://rozetka.com.ua/computers-notebooks/c80253/').start();
