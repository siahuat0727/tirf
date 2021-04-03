import cv2
import numpy as np
import json
import time
from itertools import count

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

def find_connected_component(graph, frame):
    assert graph.ndim == 2
    m, n = graph.shape
    vis = set()

    xs, ys = graph.nonzero()
    nonzero = set(zip(
        xs.astype(np.int32).tolist(), ys.astype(np.int32).tolist()))

    def dfs(xy):
        res = set()
        if xy in vis:
            return res
        if xy not in nonzero:
            return res
        vis.add(xy)
        x, y = xy

        return {xy}.union(*[
            dfs(xy)
            for xy in [(x+1, y), (x-1, y), (x, y-1), (x, y+1)]
        ])

    return [
        ret
        for xy in nonzero
        if (ret := dfs(xy))
    ]


def process(cons, args):
    total_point = sum(len(con) for _, con in cons)
    middle_frame = sum(frame * len(con) for frame, con in cons) // total_point
    xs, ys = zip(*[
        (x, y)
        for _, con in cons
        for x, y in con
    ])
    info = {
        'frame_start': cons[0][0],
        'frame_end': cons[-1][0],
        'frame': middle_frame,
        'x_min': min(xs),
        'x_max': max(xs),
        'x': sum(xs) // total_point,
        'y_min': min(ys),
        'y_max': max(ys),
        'y': sum(ys) // total_point,
        'total_point': total_point,
    }
    for k in info.keys():
        if k in ('x', 'x_min', 'x_max'):
            info[k] += args.x[0]
        if k in ('y', 'y_min', 'y_max'):
            info[k] += args.y[0]
    print(f'\r{total_point=}')
    return info


def play(args):
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f'Check if the video path is correct: {args.video}'

    components = []
    prev_comps = []
    outlier = set()
    for frame_i in count(0):  # Infinite loop
        ret, frame = cap.read()
        if not ret:
            break

        print(f'\r{frame_i} ', end='')
        if args.start is not None and frame_i < args.start:
            continue

        if args.fps is not None:
            time.sleep(1/args.fps)

        assert frame.ndim == 3, f'Espected video in 3 chans but {frame.ndim}'
        x1, x2 = args.x
        y1, y2 = args.y
        frame = frame[x1:x2, y1:y2, 0]  # TODO: use args

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        comps = find_connected_component(fgmask, frame_i)

        prev_cur = [
            [
                i
                for i, con in enumerate(comps)
                if con & prev_cons[-1][1]
            ]
            for prev_cons in prev_comps
        ]

        for curs, p_comps in zip(prev_cur, prev_comps):
            if curs:  # If prev
                cur = comps[curs[0]]
                for cur_i in curs[1:]:
                    print('\rWarning: one to many')
                    outlier.add(id(comps))
                    cur |= comps[cur_i]
                p_comps.append((frame_i, cur))
            else:
                assert isinstance(p_comps, list), p_comps
                assert all(
                    isinstance(con, tuple) and len(con) == 2
                    for con in p_comps
                ), p_comps
                components.append(process(p_comps, args))

        prev_comps = [
            con
            for curs, con in zip(prev_cur, prev_comps)
            if curs
        ]

        cur_connected = set(
            i
            for ll in prev_cur
            for i in ll
        )

        if len(cur_connected) != len(
                list(i for ll in prev_cur for i in ll)):
            print('\rWarning: many to one')

        for i, con in enumerate(comps):
            if i not in cur_connected:
                prev_comps.append([(frame_i, con)])

        # fgmask = np.stack([fgmask for _ in range(3)], axis=2)
        # frame = cv2.hconcat([frame, fgmask])
        # cv2.imshow('window-name', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    with open(args.json, 'w') as f:
        json.dump(components, f)
    print(f'\nFound {len(components)} events, save to {args.json}')

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows
