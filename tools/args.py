import argparse

def para_parser():
    arg_parser = argparse.ArgumentParser(description='MDvsFA')
    arg_parser.add_argument('--epochs', default=30)
    arg_parser.add_argument('--batch-size', default=10)
    arg_parser.add_argument('--parallel', default=True)
    arg_parser.add_argument('--d-lr', default=1e-5)
    arg_parser.add_argument('--g1-lr', default=1e-4)
    arg_parser.add_argument('--g2-lr', default=1e-4)
    arg_parser.add_argument('--lambda1', default=100)
    arg_parser.add_argument('--lambda2', default=1)
    arg_parser.add_argument('--training-imgs', default='./data/training/*_1.png')
    arg_parser.add_argument('--training-masks', default='./data/training/*_2.png')
    arg_parser.add_argument('--evl-imgs', default='./data/test_org/*')
    arg_parser.add_argument('--evl-masks', default='./data/test_gt/*')
    arg_parser.add_argument('--is-anm', default=False)
    args = arg_parser.parse_args()
    return args