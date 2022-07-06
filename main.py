from argparse import ArgumentParser
from utils import play
from utils.generate import generate


def parse():
    parser = ArgumentParser()

    parser.add_argument('--task', type=str,
                        choices=['play', 'gen'])

    parser.add_argument('--fps', type=float, default=None,
                        help='Frames per second')

    # Input and Output
    parser.add_argument('--input', type=str,
                        help='input path')
    parser.add_argument('--input_type', type=str,
                        choices=['video', 'nd2', 'images'],
                        default='video',
                        help='images: image directory with only images '
                             '(can be directly sorted)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output results path')
    parser.add_argument('--pkl', type=str, default='info.pkl',
                        help='Pickle file (input for GEN)')
    parser.add_argument('--cc', type=str, default='cc.pkl',
                        help='Path to save connected components '
                             'file (input for GEN)')

    # PLAY mode setting
    parser.add_argument('--start', type=int, default=None,
                        help='In PLAY mode, skip until this frame')
    parser.add_argument('--x', type=int, nargs='+',
                        help='In PLAY mode, the range of x-axis displayed')
    parser.add_argument('--y', type=int, nargs='+',
                        help='In PLAY mode, the range of y-axis displayed')

    parser.add_argument('--reverse', action='store_true',
                        help='Whether process the video reversely')
    parser.add_argument('--show', action='store_true',
                        help='')
    parser.add_argument('--shows', type=int, nargs='+',)

    # GEN mode setting
    parser.add_argument('--n_frame', type=int, default=60)
    parser.add_argument('--remove', choices=['top_right', 'top_left'])
    parser.add_argument('--rm_x', type=int, help='remove x threshold')
    parser.add_argument('--rm_y', type=int, help='remove y threshold')


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
    match args.task:
        case 'play':
            play(args)
        case 'gen':
            generate(args)


main()
