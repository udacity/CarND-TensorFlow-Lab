import os

from quiz_env.boilerplate.all import run_quiz_task
from quiz_env.a_tensorflow_input.quiz import run as run_quiz_a
from quiz_env.a_tensorflow_input.quiz_test import get_result as get_result_a
from quiz_env.b_tensorflow_math.quiz import run as run_quiz_b
from quiz_env.b_tensorflow_math.quiz_test import get_result as get_result_b

quizzes = [(run_quiz_a, get_result_a), (run_quiz_b, get_result_b)]


def test_quizzes(task_type):
    """
    Test all quizzes
    """
    for q, q_test in quizzes:
        print(run_quiz_task(q, q_test, task_type))


if __name__ == '__main__':
    test_quizzes('exec')
