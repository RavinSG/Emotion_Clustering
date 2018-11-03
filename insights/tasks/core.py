#
# Core Task Functions and classes live here.
#


import os
import time
import pymongo

import config


class TaskScheduler:
    def __init__(self):
        self.tasks = {}
        self.running = False

    def start(self):
        client = pymongo.MongoClient(config.MONGO_HOST, config.MONGO_PORT)
        db = client[config.MONGO_DBNAME]
        tasks_DB = db.tasks
        self.running = True
        while self.running:
            cursor = tasks_DB.find({'status': 'Added'})
            for r in cursor:
                result = self.tasks[r['type']](db, r['data'])
                tasks_DB.find_one_and_update(
                    {"_id": r['_id']}, {"$set": {"result": result, 'status': 'Finished'}})
                print('Task {} completed.'.format(r['_id']))
            time.sleep(10)

    def stop(self):
        self.running = False
