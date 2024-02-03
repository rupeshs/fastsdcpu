# Why does this file exist, and why not put this in `__main__`?
#
# You might be tempted to import things from `__main__` later,
# but that will cause problems: the code will get executed twice:
#
# - When you run `python -m super_image` python will execute
#   `__main__.py` as a script. That means there won't be any
#   `super_image.__main__` in `sys.modules`.
# - When you import `__main__` it will get executed again (as a module) because
#   there's no `super_image.__main__` in `sys.modules`.

"""Module that contains the command line application."""

import argparse
from typing import List, Optional


def get_parser() -> argparse.ArgumentParser:
    """
    Return the CLI argument parser.

    Returns:
        An argparse parser.
    """
    return argparse.ArgumentParser(prog="super-image")


def main(args: Optional[List[str]] = None) -> int:
    """
    Run the main program.

    This function is executed when you type `super-image` or `python -m super_image`.

    Arguments:
        args: Arguments passed from the command line.

    Returns:
        An exit code.
    """
    parser = get_parser()
    opts = parser.parse_args(args=args)
    print(opts)  # noqa: WPS421 (side-effect in main is fine)
    return 0
