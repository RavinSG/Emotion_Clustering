"""
Settings for Eve for connection with MongoDB
"""
import config

SERVER_NAME = config.SERVER_NAME
MONGO_HOST = config.MONGO_HOST
MONGO_PORT = config.MONGO_PORT

MONGO_DBNAME = config.MONGO_DBNAME

X_DOMAINS = '*'
X_HEADERS = ['Authorization', 'Content-type']

corpus = {
    'item_title': 'corpus',
    'additional_lookup': {
        'url': 'regex("[\w]+")',
        'field': 'id'
    },
    'resource_methods': ['GET', 'POST'],
    'schema': {
        'id': {
            'type': 'string'
        },
        'name': {
            'type': 'string'
        },
        'tweets': {
            'type': 'list',
            'schema': {
                'type': 'objectid',
                'data_relation': {
                    'resource': 'tweets',
                    'field': '_id',
                    'embeddable': True
                }
            }
        }
    }
}

tweets = {
    'item title': 'tweets',
    'schema': {
        'id': {  # TweetID from twitter
            'type': 'string'
        },
        'tweet': {
            'type': 'string'
        }
    }
}

tasks = {
    'item title': 'tasks',
    'resource_methods': ['GET', 'POST'],
    'schema': {
        'type': {
            'type': 'number'
        },
        'name': {
            'type': 'string',
        },
        'status': {
            'type': 'string'
        },
        'data': {
            'type': 'dict'
        } 
        # ,
        # 'results': {
        #     'type': 'list',
        #     'schema': {
        #         'type': 'number'
        #     }
        # }
    }
}

DOMAIN = {
    'corpus': corpus,
    'tweets': tweets,
    'tasks': tasks,
}
