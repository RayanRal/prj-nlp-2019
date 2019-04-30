const fs = require('fs');
const path = require('path');
const fetch = require('node-fetch');
const querystring = require('querystring');

 const getData = async () => {
  const endpoint = 'https://dbpedia.org/sparql?';
  const query = `
    PREFIX lng: <http://purl.org/linguistics/gold/>
    SELECT ?countryLabel ?capitalLabel ?wiki
    WHERE {
      ?country a dbo:Country ; rdfs:label ?countryLabel .
      ?country foaf:isPrimaryTopicOf	?wiki .
      ?country dbo:capital ?capital .
      ?capital rdf:type dbo:City ; rdfs:label ?capitalLabel .
      
      FILTER langMatches(lang(?capitalLabel),'en') .
      FILTER langMatches(lang(?countryLabel),'en') .
      
      FILTER EXISTS { ?country dbo:areaTotal ?area } .
      MINUS { ?country lng:hypernym dbr:Province }
      MINUS { ?country lng:hypernym dbr:Region }
      MINUS { ?country dbo:dissolutionYear ?yearEnd } .
      MINUS { ?country dct:subject dbc:Unrecognized_or_largely_unrecognized_states } .
    }
  `
  const params = {
    'default-graph-uri': 'http://dbpedia.org',
    query,
    format: 'application/sparql-results+json',
    CXML_redir_for_subjs: 121,
    CXML_redir_for_hrefs: '',
    timeout: 30000,
    debug: 'on',
    run: 'Run Query',
  };
  const qs = querystring.stringify(params);

  return fetch(endpoint + qs).then(res => res.json()).then(res => res.results.bindings);
};

(async () => {
  const rawData = await getData();
  const outputPath = path.join('..', 'data', 'countries.tsv');

  const flattenedData = rawData.reduce((acc, curr) => {
    const { countryLabel, capitalLabel, wiki } = curr;
    const [country, capital, url] = [countryLabel.value, capitalLabel.value, wiki.value];

    acc[country] = acc[country] || {};
    const currCountry = acc[country]

    currCountry.capital = currCountry.capital || new Set();
    currCountry.capital.add(capital.split(',')[0].replace(/\(.*\)/g, '').trim());
    currCountry.url = currCountry.url || url;

    return acc;
  }, {});

  const data = Object.entries(flattenedData).map(([key, { capital, url }]) => {
    return [url, key, JSON.stringify(Array.from(capital))].join('\t')
  });
  fs.writeFileSync(outputPath, data.join('\n'));
})();