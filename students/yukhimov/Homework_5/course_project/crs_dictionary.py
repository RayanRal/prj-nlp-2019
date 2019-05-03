from bs4 import BeautifulSoup
import requests

response = requests.get('http://happy2movelondon.co.uk/complete-dictionary-of-cockney-rhyming-slang/')
soup = BeautifulSoup(response.text, "html.parser")

slang = soup.find_all('h2', {'class': 'term-title'})
words = soup.find_all('p', {'class': 'term-description'})

with open('CRS_Dictionary.txt', "w") as output_file:
    for index in range(len(words)):
        output_file.write('{} - {}\n'.format(words[index].text, slang[index].text))
