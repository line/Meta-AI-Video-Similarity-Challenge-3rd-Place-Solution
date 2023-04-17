def pytest_addoption(parser):
    parser.addoption("--submission-path", action="store", default="submission.csv")
