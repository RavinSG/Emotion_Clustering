import pprint
from cerberus import Validator
import addTweets

schema = {
    'corpus_id':{
        'type': 'string'
    },

    'tweet_list': {
        'type': 'list',
        'schema': {
            'type': 'dict',
            'schema': {
                'id': {
                    'type': 'string'
                },
                'tweet': {
                    'type': 'string'
                }
            }

        }
    }
}

jobs = {
    'corpus': {
        'type': 'string'
    },
    'features': {
        'type': 'list'
    },
    'status': {
        'type': 'string'
    }
}

#doc = addTweets.add('sample_data.csv')
v = Validator(jobs)
test = {
    'corpus': "Tweets from SL",
    'features': ['one','two'],
    'status': 'Added'
}
pprint.pprint(test)
print(v.validate(test))
print(v.errors)
