import scrapy
import pickle
from bs4 import BeautifulSoup


class RozetkaSpider(scrapy.Spider):
    name = "rozetka"
    base_url = 'https://rozetka.com.ua/'
    visited = dict()  # key -> product id, value - set of visited comment pages

    def start_requests(self):
        page_url = self.base_url + 'mobile-phones/c80003/page={};preset=smartfon;view=list/'
        for i in range(1, 100):
            yield scrapy.Request(url=page_url.format(i), callback=self.parse)

    def parse(self, response):
        page = BeautifulSoup(response.body, 'html.parser')
        result = []
        for goods in page.find_all(attrs={"class": "g-i-list-title"}):
            product_link = goods.find('a').attrs['href']
            product_link = product_link + 'comments/'
            result.append(scrapy.Request(product_link, callback=self._parse_product_comments))

        return result

    def _parse_product_comments(self, response):
        page = BeautifulSoup(response.body, 'html.parser')

        self._add_visited_page(response.url)

        next_comments = []
        for comment_page in page.find_all(attrs={"class": "paginator-catalog-l-link"}):
            if comment_page.attrs.get('href'):
                next_page_url = comment_page.attrs['href']
                product_id = next_page_url.split('/')[4]
                page_number = next_page_url.split('/')[6].replace('page=', '')

                # check visited doesn't contain page before adding
                if (not self.visited.get(product_id)) or (self.visited.get(product_id)
                                                          and (page_number not in self.visited[product_id])):
                    next_comments.append(scrapy.Request(next_page_url, callback=self._parse_product_comments))

        parsed_comments = []

        for comment in page.find_all("article", {"class": "pp-review-i"}):
            text = comment.find("div", {"class": "pp-review-text-i"}).text

            useful = comment.find("div", {"class": "pp-comments-author-good-vote"})
            if useful:
                useful = useful.text.strip().rstrip('% пользователей считают этот отзыв полезным')

            rating = comment.find("span", {"class": "sprite g-rating-stars-i"})
            if rating:
                rating = rating.attrs['content']

            if text:
                parsed_comments.append(dict(text=text, rating=rating, useful=useful))

        self._save_comments(parsed_comments)
        return next_comments

    def _save_comments(self, comments):
        with open('comments.p', 'ab') as fp:
            pickle.dump(comments, fp)

    def _add_visited_page(self, current_url):
        product_id = current_url.split('/')[4]
        page_number = current_url.split('/')[6].replace('page=', '')

        if self.visited.get(product_id):
            self.visited[product_id].add(page_number)
        else:
            self.visited[product_id] = set()
            self.visited[product_id].add(page_number)
