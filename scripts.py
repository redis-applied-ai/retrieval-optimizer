import subprocess


def load_data():
    print("")


def start_app():
    # start app
    subprocess.run(
        ["uvicorn", "label_app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )


def format():
    subprocess.run(["isort", ".", "--profile", "black"], check=True)
    subprocess.run(["black", "."], check=True)


def check_format():
    subprocess.run(["black", "--check", "."], check=True)


def sort_imports():
    subprocess.run(["isort", ".", "./tests/", "--profile", "black"], check=True)


def check_sort_imports():
    subprocess.run(["isort", ".", "--check-only", "--profile", "black"], check=True)


def check_lint():
    subprocess.run(["pylint", "--rcfile=.pylintrc", "."], check=True)


def mypy():
    subprocess.run(["python", "-m", "mypy", "."], check=True)


def test():
    subprocess.run(["python", "-m", "pytest", ".", "--log-level=CRITICAL"], check=True)


def test_cov():
    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-vv",
            "--cov=.",
            "--cov-report=xml",
            "--log-level=CRITICAL",
        ],
        check=True,
    )


def cov():
    subprocess.run(["coverage", "html"], check=True)
    print("If data was present, coverage report is in ./htmlcov/index.html")
