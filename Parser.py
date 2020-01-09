

class Parser:
    """
    """

    def __init__(self) -> object:
        """Initializing
        """
        self.features = {}
        self.structured_params = ["description", "url"]
        self.unstructured_params = ["image", "detailedDescription"]

    def parse(self, graph_object):
        """
        :param graph_object
        :returns features
        """
        features = {}
        for element in graph_object['itemListElement']:
            # print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')
            for key, value in element['result'].items():
                if key in self.structured_params:
                    features[key] = value
                elif key in self.unstructured_params:
                    if "detailedDescription" == key:
                        features[key] = value['articleBody']
                # else:
                #     print("New key: ", key, " Value: ", value)
            # get the first element (with highest score)
            # Todo optimize this step of stopping
            break

        return features

    def structured(self, key, value):
        """ Handling structured params
        :param key
        :param value
        """
        self.features[key] = value

    def unstructured(self, key, value):
        """ Handling unstructured params
        :param key
        :param value
        """
        if "detailedDescription" == key:
            self.features[key] = value['articleBody']
