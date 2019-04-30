from typing import List, Dict

import re
import scrapy
from scrapy.http import HtmlResponse


products = {
    'blender': 'https://bt.rozetka.com.ua/bosch_msm2620b/p15952387/comments/',
    'mixer': 'https://bt.rozetka.com.ua/16634/p16634/comments/',
}


class RozetkaSpider(scrapy.Spider):
    name = 'rozetka'

    def start_requests(self):
        product = getattr(self, 'product', 'blender')
        yield scrapy.Request(products[product], self.parse)

    def parse(self, response):
        selector = response.css('ul[name=paginator] li::attr(id)')

        result = list()
        if selector:
            n_comment_pages = int(selector[-1].get().split(' ')[-1])
            for i in range(1, n_comment_pages + 1):
                x = scrapy.Request(f'{response.url}/page={i}', callback=self.parse_comment_page)
                result.append(x)
        return result

    def parse_comment_page(self, response: HtmlResponse):
        comments = list()
        for comment in response.css('div[id=comments] article'):
            date = comment.css('meta[itemprop=datePublished]::attr(content)').get()
            author = comment.css('span[itemprop=author]::text').get().strip()
            rating = comment.css('div[class=g-rating-b] span span::attr(content)').get()
            likes = comment.css('span[name=positive_block] span[name=count]::text').get()
            dislikes = comment.css('span[name=negative_block] span[name=count]::text').get()

            likes = int(likes) if likes else 0
            dislikes = int(dislikes) if dislikes else 0

            if rating:
                rating = int(rating)

            text = list()
            div_texts = comment.css('div[class=pp-review-text-i]')
            for dt in div_texts:
                dt = dt.css('::text').extract()
                text.append(' '.join([x.strip() for x in dt]).strip())
            text = '\n'.join(text)

            if text.strip():
                comments.append(dict(date=date, author=author, rating=rating,
                                     text=text, likes=likes, dislikes=dislikes))

        return comments



CATEGORIES = {
    'tehnika-dlya-kuhni': 'https://rozetka.com.ua/tehnika-dlya-kuhni/c435974/',
    'mebel': 'https://rozetka.com.ua/mebel/c152458/',
    'kupanie-i-gigiena': 'https://rozetka.com.ua/kupanie-i-gigiena/c3973660/',
}


class FullRozetkaSpider(scrapy.Spider):
    name = 'full'

    start_urls = [
        'https://rozetka.com.ua/kupanie-i-gigiena/c3973660/'
    ]

    # def start_requests(self):
    #     category = getattr(self, 'category', 'tehnika-dlya-kuhni')
    #     if category in CATEGORIES:
    #         yield scrapy.Request(CATEGORIES[category], callback=self.parse)
    #     else:
    #         self.log(f'NO SUCH CATEGORY {self.category}')

    def parse(self, response: HtmlResponse):
        # self.log('IN PARSE')
        subcats = self.get_subpages_links(response)
        # self.log(subcats)
        for subcat in subcats:
            # self.log(subcat)
            yield scrapy.Request(subcat, callback=self.sub_categories)

    def sub_categories(self, response: HtmlResponse):
        # self.log('IN SUBCATEGORY')
        tables = response.css('div[class=pab-table]')
        for table in tables:
            for link in set(table.css('::attr(href)').extract()):
                yield scrapy.Request(f'{link}/filter', callback=self.products_grid)

    def products_grid(self, response: HtmlResponse):
        pages = self.get_subpages_links(response)
        for page in pages:
            yield scrapy.Request(page, callback=self.parse_product)

    def parse_product(self, response: HtmlResponse):
        links = response.css('div[class=g-i-tile-i-box-desc] div[name=prices_active_element_original]')
        for link in links.css('div[class=g-rating] a::attr(href)').getall():
            yield scrapy.Request(link, callback=self.parse_comment_pages)

    def parse_comment_pages(self, response: HtmlResponse):
        subpages = self.get_subpages_links(response)
        for subpage in subpages:
            yield scrapy.Request(subpage, callback=self.parse_comment_page)

    def parse_comment_page(self, response: HtmlResponse):
        comments = list()
        for comment in response.css('div[id=comments] article'):
            date = comment.css('meta[itemprop=datePublished]::attr(content)').get()
            author = comment.css('span[itemprop=author]::text').get().strip()
            rating = comment.css('div[class=g-rating-b] span span::attr(content)').get()
            likes = comment.css('span[name=positive_block] span[name=count]::text').get()
            dislikes = comment.css('span[name=negative_block] span[name=count]::text').get()

            likes = int(likes) if likes else 0
            dislikes = int(dislikes) if dislikes else 0

            if rating:
                rating = int(rating)

            text = list()
            div_texts = comment.css('div[class=pp-review-text-i]')
            for dt in div_texts:
                dt = dt.css('::text').extract()
                text.append(' '.join([x.strip() for x in dt]).strip())
            text = '\n'.join(text)

            if text.strip():
                comments.append(dict(date=date, author=author, rating=rating,
                                     text=text, likes=likes, dislikes=dislikes))

        return comments

    def get_subpages_links(self, response: HtmlResponse):
        links = [response.url]
        selector = response.css('ul[name=paginator] li::attr(id)')
        if selector:
            match = re.match(r'page\s?(?P<num>\d+)', selector[-1].get())
            n_pages = int(match.group('num'))
            # n_pages = int(re.split(r'page\s?', selector[-1].get())[-1])
            for i in range(2, n_pages + 1):
                links.append(f'{response.url}/page={i}')
        return links


