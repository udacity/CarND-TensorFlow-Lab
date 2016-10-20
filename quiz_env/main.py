from quiz_env.boilerplate.all import run_quiz_task
from quiz_env.a_tensorflow_input.quiz import run as run_quiz_a
from quiz_env.a_tensorflow_input.quiz_test import get_result as get_result_a
from quiz_env.b_tensorflow_math.quiz import run as run_quiz_b
from quiz_env.b_tensorflow_math.quiz_test import get_result as get_result_b
from quiz_env.c_tensorflow_linear_regression_function_a.quiz import run as run_quiz_c_a
from quiz_env.c_tensorflow_linear_regression_function_a.quiz_test import get_result as get_result_c_a
from quiz_env.c_tensorflow_linear_regression_function_b.quiz import run as run_quiz_c_b
from quiz_env.c_tensorflow_linear_regression_function_b.quiz_test import get_result as get_result_c_b
from quiz_env.d_tensorflow_softmax.quiz import run as run_quiz_d
from quiz_env.d_tensorflow_softmax.quiz_test import get_result as get_result_d
from quiz_env.e_tensorflow_cross_entropy.quiz import run as run_quiz_e
from quiz_env.e_tensorflow_cross_entropy.quiz_test import get_result as get_results_e

quizzes = [('A', run_quiz_a, get_result_a), ('B', run_quiz_b, get_result_b), ('C-A', run_quiz_c_a, get_result_c_a),
           ('C - B', run_quiz_c_b, get_result_c_b), ('D', run_quiz_d, get_result_d), ('E', run_quiz_e, get_results_e)]


def test_quizzes(task_type):
    """
    Test all quizzes
    """
    for name, q, q_test in quizzes:
        print('{}:'.format(name))
        print(run_quiz_task(q, q_test, task_type))


if __name__ == '__main__':
    test_quizzes('exec')
