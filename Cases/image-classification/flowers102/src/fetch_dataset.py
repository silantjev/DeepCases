import argparse
from utils.flower_data_manager import fetch_flowers_by_torch

parser = argparse.ArgumentParser(description='Скачать данные Oxford 102 Flower', add_help=False) 
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='показать справку и выйти')
parser.add_argument('--dir', type=str, default=None, help='папка для данных (по умолчанию "data/")')
args = parser.parse_args()

fetch_flowers_by_torch(data_dir=args.dir)
