import re
import requests

from scrapy.http import HtmlResponse
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor


class ForumSpider(CrawlSpider):
    name = 'forum'
    allowed_domains = ['forum.lvivport.com']
    start_urls = ['http://forum.lvivport.com/forums/velodorizhka.92/']
    rules = [
        Rule(LinkExtractor(allow=r'forums/velodorizhka.92/page-\d+$'), callback='parse_room', follow=True),
    ]

    def parse_room(self, response):
        for theme in response.css('li.discussionListItem'):
            thread_link = theme.css('div.titleText a.PreviewTooltip::attr(href)').get(),
            thread_link = f'http://{self.allowed_domains[0]}/{thread_link[0]}'
            yield {
                'title': theme.css('div.titleText a::text').get(),
                'date': theme.css('span.DateTime::text').get(),
                'n_messages': theme.css('div.listBlock dl.major dd::text').get(),
                'n_views': theme.css('div.listBlock dl.minor dd::text').get(),
                'thread' : self.parse_thread(thread_link)
            }

    def parse_thread(self, link):
        thread = list()
        response = HtmlResponse(body=requests.get(link).content, url=link)
        page_links = response.css('div.PageNav a::attr(href)').getall()
        if not page_links:
            return self.parse_thread_page(link)

        for page_link in page_links:
            page_link = f'http://{self.allowed_domains[0]}/{page_link}'
            thread_page = self.parse_thread_page(page_link)
            thread.extend(thread_page)
        return thread

    def parse_thread_page(self, link):
        thread_page = HtmlResponse(body=requests.get(link).content, url=link)
        return [self.parse_message(msg) for msg in thread_page.css('ol.messageList li.message')]

    def parse_message(self, msg):
        user = msg.css('div.messageUserBlock')
        content = msg.css('div.messageContent')
        likes = msg.css('div.secondaryContent')

        return {
            'user': self.parse_user(user),
            'likes': self.parse_likes(likes),
            'texts': self.parse_texts(content),
            'quotes': self.parse_quotes(content),
            'date': msg.css('div.messageMeta span.DateTime::attr(title)').get()
        }

    def parse_texts(self, msg):
        msg = msg.css('article')
        texts = (x.strip() for x in msg.css('::text').extract())
        texts = filter(lambda x: bool(x) and re.search(r'[а-яА-Я]+', x), texts)
        texts = re.split(r'Цитата від.*?Натисніть, щоб розгорнути...', ' '.join(texts))
        return [x.strip() for x in texts if x.strip()]

    def parse_quotes(self, msg):
        quotes = list()
        for quote in msg.css('div.bbCodeQuote'):
            quotes.append({
                'user_name': quote.css('::attr(data-author)').get(),
                'text': quote.css('div.quote::text').get()
            })
        return quotes

    def parse_user(self, user):
        return {
            'name': user.css('a.username::text').get(),
            'title': user.css('em.userTitle::text').get()
        }

    def parse_likes(self, likes):
        result = list()
        for like in likes.css('li'):
            result.append({
                'type': like.css('img::attr(title)').get(),
                'count': int(like.css('strong::text').get())
            })
        return result
