from argparse import ArgumentParser
from utils.play import play
from utils.generate import generate


def parse():
    parser = ArgumentParser()

    parser.add_argument('--task', type=str,
                        choices=['play', 'gen'])
    parser.add_argument('--fps', type=int, default=None,
                        help='Frames per second')
    parser.add_argument('--start', type=int, default=None,
                        help='In PLAY mode, skip until this frame')
    parser.add_argument('--video', type=str,
                        help='Input video path')
    parser.add_argument('--x', type=int, nargs='+',
                        help='In PLAY mode, the range of x-axis displayed')
    parser.add_argument('--y', type=int, nargs='+',
                        help='In PLAY mode, the range of y-axis displayed')
    parser.add_argument('--json', type=str, default='info.json',
                        help='Json file (output for DECT, input for GEN)')

    args = parser.parse_args()
    assert args.fps is None or args.fps > 0, 'fps must > 0'
    assert args.x is None or (len(args.x) == 2 and
                              args.x[0] < args.x[1]), 'Usage (eg.): --x 50 150'
    assert args.y is None or (len(args.y) == 2 and
                              args.y[0] < args.y[1]), 'Usage (eg.): --y 50 150'
    # TODO: save info file
    return args


def main():
    args = parse()
    if args.task == 'play':
        play(args)
    elif args.task == 'gen':
        generate(args)


main()
