import numpy as np


def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = 0.356675
    result = {
        'correct': False,
        'feedback': '',
        'comment': ''}

    output = student_func()
    if not isinstance(output, np.float32):
        result['feedback'] = 'Output is the wrong type.'
        result['comment'] = 'The output should come from running the session.'
    elif np.allclose(output, answer):
        result['correct'] = True
        result['feedback'] = 'Your Cross Entropy function is correct'

    return result
