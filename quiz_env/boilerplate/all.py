from argparse import ArgumentParser
import json
import os


def run_python3_shell(script_path):
    """
    Run the <script_path> script in python 3
    """
    os.system('python3 ' + script_path)


def run_quiz_task(student_func, test_func, task_type):
    output = None

    if task_type == 'test':
        # Test run of student's code.  Print out the results.
        output = student_func()
    elif task_type == 'exec':
        # Print JSON results of running unit tests against <student_func>
        # The JSON is has the keys 'correct', 'feedback', and 'comment'
        try:
            result = test_func(student_func)
        except Exception as err:
            # Default error result
            result = {
                'correct': False,
                'feedback': 'Something went wrong with your submission:',
                'comment': str(err)}
        
        output = json.dumps(result)

    return output


if __name__ == '__main__':
    # Import quiz and quiz_test in python 3 only
    from quiz import run
    from quiz_test import get_result

    arg_parser = ArgumentParser()
    arg_parser.add_argument('TaskType', help='The task to run', choices=['test', 'exec'])
    args = arg_parser.parse_args()

    print(run_quiz_task(run, get_result, args.TaskType))
