from contextlib import contextmanager

from .utils import Singleton

LINE_WIDTH = 110
REPORT = None
LOG_HOOK = print


def log(string):
    LOG_HOOK(string)


class Report:
    def __init__(self, file_path=None):
        self.lines = []
        self.file_path = file_path

    def to_str(self):
        return "\n".join(self.lines)

    def get_hook(self):
        def hook(string):
            self.lines.append(string)

        return hook

    def save(self):
        if self.file_path:
            with open(self.file_path, "w") as f:
                f.write(self.to_str())


@contextmanager
def ReportGuard(report):
    global LOG_HOOK, REPORT
    try:
        old_REPORT = REPORT
        old_log = LOG_HOOK
        REPORT = report
        if REPORT is not None:
            LOG_HOOK = report.get_hook()
        yield
    finally:
        if REPORT is not None:
            REPORT.save()
            LOG_HOOK = old_log
        REPORT = old_REPORT


def create_title(title, mark="="):
    name_len = (len(title) + 6) // 2 * 2
    format_str = "{left}{title:^" + str(name_len) + "s}{right}"
    title_str = format_str.format(
        left=mark * ((LINE_WIDTH - name_len) // 2),
        title=title,
        right=mark * ((LINE_WIDTH - name_len) // 2),
    )
    return title_str


@contextmanager
def ReportTitle(title, mark="="):
    log(create_title(title, mark))
    yield
    log(mark * LINE_WIDTH + "\n")
