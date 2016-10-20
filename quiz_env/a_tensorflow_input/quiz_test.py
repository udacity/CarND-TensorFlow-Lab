from tensorflow.python.framework.errors import InvalidArgumentError
import numpy as np

def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = 123
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    try:
        output = student_func()
        if not output:
            result['feedback'] = 'No output found'
        if not isinstance(output, np.ndarray):
            result['feedback'] = 'Output is the wrong type.'
            result['comment'] = 'The output should come from running the session.'
        if output == answer:
            result['correct'] = True
            result['feedback'] = 'You got it right.  You figure out how to use feed_dict!'
    except InvalidArgumentError as err:
        if err.message.startswith('You must feed a value for placeholder tensor'):
            result['feedback'] = 'The placeholder is not being set.'
            result['comment'] = 'Try using the feed_dict.'
        else:
            raise

    return result
