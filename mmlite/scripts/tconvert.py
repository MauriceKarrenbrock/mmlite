# -*- coding: utf-8 -*-
"""
Convert a trj from/to various formats.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from mmlite.utils import convert_trajectory


def parse_args():
    """Parse command line options."""
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description=__doc__)
    # positional args
    parser.add_argument('trj_in', type=str)
    parser.add_argument('trj_out', type=str)
    parser.add_argument('--start',
                        type=int,
                        help='Start index for slicing.',
                        default=0)
    parser.add_argument('--stop', type=int, help='Stop index for slicing.')
    parser.add_argument('--step',
                        type=int,
                        help='Step index for slicing.',
                        default=1)
    parser.add_argument('-t', '--topology', type=str)
    parser.add_argument('--split', action='store_true', default=False)
    return parser.parse_args()


def main():
    """Call convert_trajectory."""
    kwargs = vars(parse_args())
    # print(kwargs)
    convert_trajectory(**kwargs)


if __name__ == '__main__':
    main()
