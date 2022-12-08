import sys
import math


def progressbar(status, total, scale=20):
    cnt = math.ceil(status / total * scale)
    return f"[{''.join(['='] * cnt)}{''.join([' '] * (scale - cnt))}]"


class ProgressbarUnit(object):
    def __init__(self, total: int, scale: int, header: str='', show_percent: bool=False, hide_after_fin: bool=True):
        self.total = total
        self.scale = scale

        self.header = header
        self.show_percent = show_percent
        self.hide_after_fin = hide_after_fin

        self.status = 0

    def reset(self):
        self.status = 0

    def trigger(self, amount=0):
        if self.status >= self.total:
            return

        self.status += amount
        sys.stdout.write(f"\r{self.header} {progressbar(status=self.status, total=self.total, scale=self.scale)}")

        if self.show_percent:
            sys.stdout.write(f" {self.status/self.total*100:.0f}")
        if self.status >= self.total:
            sys.stdout.write('\r' if self.hide_after_fin else '\n')