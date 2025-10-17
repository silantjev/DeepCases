import argparse

def make_arg_parser(description, params=False):
    parser = argparse.ArgumentParser(description=description, add_help=False) 

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='показать справку и выйти')
    if params:
        parser.add_argument('--params', type=str, default=None, help='конфигурационный yaml-файл')
    parser.add_argument('--conf', '--split-conf', type=str, default=None, help='конфигурационный json-файл с процентами')

    return parser

