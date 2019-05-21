from bs4 import BeautifulSoup
import requests

URL = 'https://www.theidioms.com/list/'

with open('idioms.txt', "w") as output_file:
    for page_number in range(1, 137):
        print(page_number)
        url = URL + 'page/{}'.format(page_number)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        lines = soup.find_all('dt')
        for link in lines:
            output_file.write(link.find('a').text + '\n')


