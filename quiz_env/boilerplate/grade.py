import json


try:
    result = json.loads(executor_result['stdout'])
except:
    result = {
        'correct': False,
        'feedback': 'There\'s an error with the code',
        'comment': 'Click "Test Run" to see the error message'}

grade_result.update(result)
