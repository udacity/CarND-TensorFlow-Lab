def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = None
    result = {
        'correct': False,
        'feedback': 'That\'s the wrong answer.  It should print {}'.format(answer),
        'comment': ''}

    output = student_func()
    # Check output

    return result
