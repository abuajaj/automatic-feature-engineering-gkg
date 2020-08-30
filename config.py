# DB
dialect = "mysql"
driver = "mysqlconnector"
host = "localhost"
port = "8889"
username = "freebase"
password = "freebase"
database = "freebase"
DB_URI = f"{dialect}+{driver}://{username}:{password}@{host}:{port}/{database}"
BASE_URL = "Freebase"

# FEATURES TUNING
THRESHOLD = 0.7
TOP_N_FEATURES = 5

# SOURCE
LOOKUP = 'lookup'
TARGET = 'target'
DATASET_DIR = '{PATH}/AFEGKG/datasets/'
SOURCE_PATH = '{PATH}/freebase-rdf-latest.gz'

ALLOWED_ENTITIES = {
    '/music/':                'music',
    '/film/':                 'film',
    '/tv/':                   'tv',
    '/location/':             'location',
    '/people/':               'people',
    '/measurement_unit/':     'measurement_unit',
    '/book/':                 'book',
    '/media_common/':         'media_common',
    '/medicine/':             'medicine',
    '/award/':                'award',
    '/biology/':              'biology',
    '/sports/':               'sports',
    '/organization/':         'organization',
    '/education/':            'education',
    '/baseball/':             'baseball',
    '/business/':             'business',
    '/imdb/':                 'imdb',
    '/government/':           'government',
    '/cvg/':                  'cvg',
    '/soccer/':               'soccer',
    '/time/':                 'time',
    '/astronomy/':            'astronomy',
    '/basketball/':           'basketball',
    '/american_football/':    'american_football',
    '/olympics/':             'olympics',
    '/fictional_universe/':   'fictional_universe',
    '/theater/':              'theater',
    '/visual_art/':           'visual_art',
    '/military/':             'military',
    '/protected_sites/':      'protected_sites',
    '/geography/':            'geography',
    '/broadcast/':            'broadcast',
    '/architecture/':         'architecture',
    '/food/':                 'food',
    '/aviation/':             'aviation',
    '/finance/':              'finance',
    '/transportation/':       'transportation',
    '/boats/':                'boats',
    '/computer/':             'computer',
    '/royalty/':              'royalty',
    '/library/':              'library',
    '/internet/':             'internet',
    '/wine/':                 'wine',
    '/projects/':             'projects',
    '/chemistry/':            'chemistry',
    '/cricket/':              'cricket',
    '/travel/':               'travel',
    '/symbols/':              'symbols',
    '/religion/':             'religion',
    '/influence/':            'influence',
    '/language/':             'language',
    '/community/':            'community',
    '/metropolitan_transit/': 'metropolitan_transit',
    '/automotive/':           'automotive',
    '/digicams/':             'digicams',
    '/law/':                  'law',
    '/exhibitions/':          'exhibitions',
    '/tennis/':               'tennis',
    '/venture_capital/':      'venture_capital',
    '/opera/':                'opera',
    '/comic_books/':          'comic_books',
    '/amusement_parks/':      'amusement_parks',
    '/dining/':               'dining',
    '/ice_hockey/':           'ice_hockey',
    '/event/':                'event',
    '/spaceflight/':          'spaceflight',
    '/zoo/':                  'zoo',
    '/meteorology/':          'meteorology',
    '/martial_arts/':         'martial_arts',
    '/periodicals/':          'periodicals',
    '/games/':                'games',
    '/celebrities/':          'celebrities',
    '/nytimes/':              'nytimes',
    '/rail/':                 'rail',
    '/interests/':            'interests',
    '/atom/':                 'atom',
    '/boxing/':               'boxing',
    '/comic_strips/':         'comic_strips',
    '/conferences/':          'conferences',
    '/skiing/':               'skiing',
    '/engineering/':          'engineering',
    '/fashion/':              'fashion',
    '/radio/':                'radio',
    '/distilled_spirits/':    'distilled_spirits',
    '/chess/':                'chess',
    '/physics/':              'physics',
    '/geology/':              'geology',
    '/bicycles/':             'bicycles',
    '/comedy/':               'comedy'
}

NOT_ALLOWED_TYPES = ['type', '/common/topic/topic_equivalent_webpage', '/type/object/key', '/wikipedia/en', '/en',
                     '/common/topic/alias', '/common/topic/official_website', '/common/topic/topical_webpage',
                     '/common/topic/webpage', '/common/topic/image', '/common/topic/webpage',
                     '/common/topic/notable_types']
