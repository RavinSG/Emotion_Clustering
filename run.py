from eve import Eve
import sys


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python run.py [tasks|api]')
        exit(-1)
    if sys.argv[1] == 'tasks':
        from insights.tasks import TaskScheduler, featurizer_task
        app = TaskScheduler()
        app.tasks[0] = featurizer_task
        app.start()
    elif sys.argv[1] == 'api':
        app = Eve()
        app.run(port=8080)
    else:
        print('Invalid option provided. Use `tasks` or `api`')
