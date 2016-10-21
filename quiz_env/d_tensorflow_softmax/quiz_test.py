import numpy as np
from tensorflow.python.framework.errors import InvalidArgumentError


def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = np.array([0.65900117, 0.24243298, 0.09856589])
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    try:
        output = student_func()
        if not isinstance(output, np.ndarray):
            result['feedback'] = 'Output is the wrong type.'
            result['comment'] = 'The output should come from running the session.'
        elif np.allclose(output, [2, 1, 0.1]):
            result['feedback'] = 'You\'re returning the logits.'
            result['comment'] = 'You need to apply softmax to the logits and return them.'
        elif np.allclose(output, answer):
            result['correct'] = True
            result['feedback'] = 'That\'s the correcct softmax!'
    except InvalidArgumentError as err:
        if err.message.startswith('You must feed a value for placeholder tensor'):
            result['feedback'] = 'The placeholder is not being set.'
            result['comment'] = 'Try using the feed_dict.'
        else:
            raise

    return result
