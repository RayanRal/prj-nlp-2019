import scrapy
from bs4 import BeautifulSoup

from scrapy.http import HtmlResponse


class RozetkaScrapper(scrapy.Spider):
    """
    Crawles comments for one goods category in site https://rozetka.com.ua
    """

    name = 'rspider'

    start_urls = ['https://rozetka.com.ua/mobile-phones/c80003/preset=smartfon/']

    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
    }

    def parse(self, response: HtmlResponse):
        if response.url in self.start_urls:
            yield from self.parse_first_category_page(response)

    def parse_first_category_page(self, response: HtmlResponse):
        """
        Method constructs and add into crawl queue urls for the list subpages
        """
        # Process first page
        yield from self.parse_for_product_urls(response)
        # Find the last page
        last_page_numbers = response.css('a.blacklink.paginator-catalog-l-link::text').getall()
        if last_page_numbers:
            try:
                last_num = int(last_page_numbers[-1])
                url = response.url
                page_param_pos = url.rfind('/', 0, -2) + 1
                pattern = url[:page_param_pos] + '<page=>' + url[page_param_pos:]
                for p in range(2, last_num + 1):
                    page_url = pattern.replace('<page=>', f'page={p};')
                    yield response.follow(page_url, callback=self.parse_for_product_urls)
            except Exception as ex:
                print(f"Exception: {ex}")

    def parse_for_product_urls(self, response: HtmlResponse):
        """
        Method retrieve from product list individual product urls
        """
        # print(f"parse_for_product_urls: {response.url}")
        for url in response.css('a.responsive-img.centering-child-img::attr(href)').getall():
            if url:
                yield response.follow(url + 'comments/', callback=self.parse_comment_tab)

    def parse_comment_tab(self, response: HtmlResponse):
        # print(f"parse_comment_tab: {response.url}")
        # process the first comments page
        yield from self.parse_extract_comments(response)
        # Extract comment pages
        # Find the last page
        last_page_numbers = response.css('a.blacklink.paginator-catalog-l-link::text').getall()
        if last_page_numbers:
            try:
                last_num = int(last_page_numbers[-1])
                url = response.url
                for p in range(2, last_num + 1):
                    page_url = url + f'page={p}/'
                    yield response.follow(page_url, callback=self.parse_extract_comments)
            except Exception as ex:
                print(f"Exception: {ex}")

    def parse_extract_comments(self, response: HtmlResponse):
        # print(response.url)
        goods_code = response.css('span.detail-code-i::text').get()
        res = []
        for s in response.css('div.pp-review-inner'):
            stars = s.css('span.sprite::attr(content)').get()
            if stars:
                stars = stars.strip()
                review_html = s.css('div.pp-review-text-i').get()
                soup = BeautifulSoup(review_html)
                review = soup.get_text().strip()
                author = s.css('span.pp-review-author-name::text').get().strip()
                permalink = s.css('a.pp-review-i-link::attr(href)').get()
                res.append(
                    {
                        'goods_code': goods_code,
                        'stars': stars,
                        'review': review,
                        'author': author,
                        'permalink': permalink,
                    })
        yield from res
