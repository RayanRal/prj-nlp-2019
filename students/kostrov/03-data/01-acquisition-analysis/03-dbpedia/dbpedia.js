const fs = require('fs');
const fetch = require('node-fetch');
const querystring = require('querystring');

const request = async (offset = 0) => {
  const endpoint = 'https://dbpedia.org/sparql?';
  const params = {
    'default-graph-uri': 'http://dbpedia.org',
    query: `select ?boxer, ?o, ?p where { ?boxer a dbo:Boxer . ?boxer ?p ?o . } limit 10000 offset ${offset}`,
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

const gatherRelations = (data, container) => {
  for (const item of data) {
    const { boxer: boxerUri, p, o } = item;
    const boxer = boxerUri.value;
    const object = o.value;
    const predicate = p.value;
    container[boxer] = container[boxer] || {};
    container[boxer][predicate] = container[boxer][predicate] || [];
    container[boxer][predicate].push(object);
  }
};

(async () => {
  const result = {};
  let response;
  let offset = 0;
  while (!response || response.length === 10000) {
    console.log(offset);
    response = await request(offset);
    gatherRelations(response, result);
    offset += 10000;
  }
  fs.writeFileSync('relations.json', JSON.stringify(result, null, 2));
})();
