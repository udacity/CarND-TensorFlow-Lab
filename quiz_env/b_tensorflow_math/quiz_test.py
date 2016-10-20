import numpy as np


def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = 4
    result = {
        'correct': False,
        'feedback': '',
        'comment': ''}

    try:
        output = student_func()
        if not output:
            result['feedback'] = 'No output found'
        if not isinstance(output, np.int32):
            result['feedback'] = 'Output is the wrong type.'
            result['comment'] = 'The output should come from running the session.'
        if output == answer:
            result['correct'] = True
            result['feedback'] = 'That\'s right!  You correctly turned the algorithm to TensorFlow'
    except TypeError as err:
        if str(err).endswith('into a Tensor or Operation.)'):
            result['feedback'] = 'TensorFlow session requires a tensor to run'
        else:
            raise

    return result
