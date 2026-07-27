"""
Tests for the main.py file

Info:
    More info about writing Tests:
    https://docs.pytest.org/en/8.2.x/#a-quick-example
"""

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pibob.main as main_file


def test_main():
    """a simple test to see if the main-function of the main-file can be called, without failing

    Hint:
        This Test will fail on purpose. Fix it to make the CI-pipeline pass ;)
    """
    main_file.main()  # just calls the main function and if no Error is thrown we are happy.

    print("Fixme! I am failing on purpose:")
    assert 2 == 2
