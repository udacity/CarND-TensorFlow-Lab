import numpy as np


def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = np.array([1, 2, 3, 4])
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    output = student_func()
    if not isinstance(output, np.ndarray):
        result['feedback'] = 'Output is the wrong type.'
        result['comment'] = 'The output should come from running the session.'
    elif np.array_equal(output, answer):
        result['correct'] = True
        result['feedback'] = 'You correctly initalized the TensorFlow Variable!'

    return result
