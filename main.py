from argparse import ArgumentParser
from utils import play, generate, crop_and_save
from pathlib import Path


def parse():
    parser = ArgumentParser()

    parser.add_argument('--task', type=str, choices=['all', 'play', 'gen', 'crop'], default='all')

    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second')

    # Input and Output
    parser.add_argument('--input', type=str,
                        help='input path')
    parser.add_argument('--output', type=str,
                        help='output path')
    parser.add_argument('--input_type', type=str,
                        choices=['video', 'nd2', 'images'],
                        default='video',
                        help='images: image directory with only images '
                             '(can be directly sorted)')
    parser.add_argument('--pkl', type=str, default='info.pkl',
                        help='Pickle file (input for GEN)')
    parser.add_argument('--cc', type=str, default='cc.pkl',
                        help='Path to save connected components '
                             'file (input for GEN)')

    # PLAY mode setting
    parser.add_argument('--start', type=int, default=None,
                        help='In PLAY mode, skip until this frame')
    parser.add_argument('--nms', type=float, default=2.0, help='NMS threshold (event distance)')
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
    parser.add_argument('--n_frame_before', type=int, default=30, help='Number of frames before the event')
    parser.add_argument('--n_frame_after', type=int, default=30, help='Number of frames after the event')
    parser.add_argument('--remove', choices=['top_right', 'top_left'])
    parser.add_argument('--rm_x', type=int, help='remove x threshold')
    parser.add_argument('--rm_y', type=int, help='remove y threshold')
    parser.add_argument('--crop_size', type=int, default=24, help='crop size')
    parser.add_argument('--for_anotation', action='store_true', help='For annotation, generate video with events concatenated')

    args = parser.parse_args()
    assert args.fps is None or args.fps > 0, 'fps must > 0'
    assert args.x is None or (len(args.x) == 2 and
                              args.x[0] < args.x[1]), 'Usage (eg.): --x 50 150'
    assert args.y is None or (len(args.y) == 2 and
                              args.y[0] < args.y[1]), 'Usage (eg.): --y 50 150'
    assert (args.x is None and args.y is None) or (args.x is not None and args.y is not None), 'Either x and y must be specified or neither'
    return args


def main():
    args = parse()

    # Create output directory with the same name as input but remove extension
    # Path(out_dir := Path(args.input).with_suffix('')).mkdir(exist_ok=True)
    out_dir = Path(args.input).with_suffix('')

    match args.task:
        case 'all':
            play(args, out_dir)
            generate(args, out_dir)
        case 'play':
            play(args, out_dir)
        case 'gen':
            generate(args, out_dir)
        case 'crop':
            crop_and_save(args, out_dir)


main()
