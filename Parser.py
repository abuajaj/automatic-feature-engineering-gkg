import json

from pyld import jsonld


class Parser:
    """
    """

    def __init__(self):
        """Initializing
        """

    def parse(self, graph_object):
        """
        """

        expanded = jsonld.expand(graph_object)

        print(json.dumps(expanded, indent=2))
