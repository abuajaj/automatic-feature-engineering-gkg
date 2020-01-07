"""Example of Python client calling Knowledge Graph Search API."""
import json
import urllib
import urllib.request
from urllib.parse import urlencode

from Parser import Parser


class Main:
    """
    """

    def __init__(self):
        """Initializing
        """
        self.api_key = open('.api_key').read()

    def build_query(self, query):
        """ Build HTTP query and send to GKG API,
            As response; return graph as json-LD object
            Arg: query(string)
        """
        service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
        params = {
            'query': query,
            'limit': 10,
            'indent': True,
            'key': self.api_key,
        }
        url = service_url + '?' + urllib.parse.urlencode(params)
        print(url)
        response = json.loads(urllib.request.urlopen(url).read())
        return response


main = Main()
graph_response = main.build_query('donald trump')
print(graph_response)

parser = Parser()
parser.parse(graph_response)
# for element in graph_response['itemListElement']:
#     print(element)
#     print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')
