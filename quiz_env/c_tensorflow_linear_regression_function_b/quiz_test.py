import numpy as np


def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = np.array([[-0.75549841, 1.97941792], [2.5510993, 0.77116388]])
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    output = student_func()
    if not isinstance(output, np.ndarray):
        result['feedback'] = 'Output is the wrong type.'
        result['comment'] = 'The output should come from running the session.'
    elif np.allclose(output, answer):
        result['correct'] = True
        result['feedback'] = 'You got it!  The logits are correct'

    return result
