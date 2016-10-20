import numpy as np


def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = np.array([0.65900117, 0.24243298, 0.09856589])
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    output = student_func()
    if not isinstance(output, np.ndarray):
        result['feedback'] = 'Output is the wrong type.'
        result['comment'] = 'The output should come from running the session.'
    if np.allclose(output, answer):
        result['correct'] = True
        result['feedback'] = 'That\'s the correcct softmax!'


    return result
