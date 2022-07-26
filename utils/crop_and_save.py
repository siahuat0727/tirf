from pathlib import Path
import cv2
import os
import shutil
import subprocess

from .readers import select_reader
from .utils import crop


def crop_and_save(args, out_dir):
    """
    Crop a video to a certain spatial region and save it to a path.
    """
    assert args.x is not None and args.y is not None

    if args.output is None:
        args.output = str(Path(args.input).with_suffix('')) + '_cropped.mp4'
        # args.output = Path(args.input).stem + '_cropped.mp4'
    print(f'Saving to {args.output}')

    generator = select_reader(args.input_type)(args.input)
    for i, frame in enumerate(generator):
        frame = crop(frame, *args.x, *args.y)
        cv2.imwrite(os.path.join(out_dir, f'{i:06d}.png'), frame)
    cmd = f'ffmpeg -r {args.fps} -i "{os.path.join(out_dir, "%06d.png")}" -vcodec libx264 "{args.output}"'
    print(f'Running: {cmd}')
    subprocess.call(cmd, shell=True)
    print(f'Removing {out_dir}')
    shutil.rmtree(out_dir)
    print(f'Done!')
