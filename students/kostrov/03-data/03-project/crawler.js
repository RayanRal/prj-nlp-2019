const cheerio = require('cheerio');
const fetch = require('node-fetch');

const HOST = 'https://en.wikipedia.org';
const URL = `${HOST}/wiki/List_of_Game_of_Thrones_characters`;

const getAliases = async (path) => {
  const html = await fetch(HOST + path).then(res => res.text());
  const $ = cheerio.load(html);
  const aliases = Array.from($('#mw-content-text tr'))
    .reduce((acc, curr) => {
      const $curr = $(curr);
      const th = $curr.find('th').text();
      if (th === 'Alias') {
        const contents = $curr.find('td');
        contents.find('br, b').remove();
        if (contents.has('ul').length) {
          contents.find('li').each((i, el) => {
            acc.push(...$(el).text().replace(/:|\(TV series\)/, '').trim()
              .split('/'));
          });
        } else {
          acc.push(contents.text());
        }
      }
      return acc;
    }, []);

  return { [$('#firstHeading').text()]: aliases.filter(Boolean) };
};

const getCharsWithAliases = async (url) => {
  const html = await fetch(url).then(res => res.text());
  const $ = cheerio.load(html);
  return $('.wikitable').first().find('tr td:nth-child(2) a')
    .map((i, el) => $(el).attr('href'))
    .get();
};

(async () => {
  const characters = await getCharsWithAliases(URL);
  const aliases = {};
  for (const path of characters) {
    Object.assign(aliases, await getAliases(path));
  }
  console.log(aliases);
})();
