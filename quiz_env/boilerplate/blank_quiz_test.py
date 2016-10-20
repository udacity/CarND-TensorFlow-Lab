def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    result = {
        'correct': False,
        'feedback': '',
        'comment': ''}

    try:
        output = student_func()
        # Check output
    except Exception as err:
        result['feedback'] = 'Something went wrong with your submission:'
        result['comment'] = str(err)

    return result
