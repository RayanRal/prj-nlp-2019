import re
from scrapy.spiders import CrawlSpider



class MozdocSpider(CrawlSpider):
    name = 'mozdoc'
    start_urls = [f'http://mozdocs.kiev.ua/likiview.php?id={i}' for i in range(3000, 3010)]

    def parse(self, response):
        info_table = self.parse_info_table(response.css('table')[4])
        instruction = response.css('div.instruction')
        instruction = ' '.join(instruction.css('::text').extract())
        yield {
            'info_table': info_table,
            'instruction': instruction
        }

    def parse_info_table(self, table):
        result = dict()
        for row in table.css('tr'):
            try:
                key, value = row.css('td')
            except ValueError:
                self.log(row.css('td').getall())
            else:
                key = key.css('span::text').get()
                value = value.css('::text').get()
                result[key] = value
        return result
