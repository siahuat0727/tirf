import cv2
import numpy as np
import pickle
import time
from utils import Tirf


def comp_reverse(comps, total_frame):
    def frame_reverse(comp):
        assert isinstance(comp, dict)

        def do_reverse(k):
            v = comp[{
                'frame_start': 'frame_end',
                'frame_end': 'frame_start',
            }.get(k, k)]
            if k == 'regions':
                assert all(
                    isinstance(con, tuple) and
                    len(con) == 2 and
                    isinstance(con[1], set)
                    for con in v
                ), v

                v = [
                    (total_frame-1-frame_i, comp)
                    for frame_i, comp in reversed(v)
                ]
                do_reverse.regions = True
            elif k in ('frame_start', 'frame_end', 'frame'):
                v = total_frame-1-v
                assert 0 <= v < total_frame
                do_reverse.frame = True
            return v

        res = {
            k: do_reverse(k)
            for k in comp.keys()
        }
        assert do_reverse.frame and do_reverse.regions
        return res

    return [
        frame_reverse(comp)
        for comp in comps
    ]


def comp_sort(comps):
    return list(sorted(comps, key=lambda d: d['frame_start']))


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
        'regions': cons,
    }
    for k in info.keys():
        if k in ('x', 'x_min', 'x_max'):
            info[k] += args.x[0]
        if k in ('y', 'y_min', 'y_max'):
            info[k] += args.y[0]
    return info


def play(args):

    components = []
    prev_comps = []
    outlier = set()
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    tirf = Tirf(args.video, args.x, args.y)

    frames = tirf.readall()
    if args.reverse:
        frames = reversed(list(frames))

    for frame_i, frame in enumerate(frames):
        print(f'\rFrame: {frame_i}, found {len(components)} events', end='')
        if args.start is not None and frame_i < args.start:
            continue

        frame = frame[:, :, 0]

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
                    print(f'\r{frame_i} - Warning: one to many')
                    outlier.add(id(comps))
                    cur |= comps[cur_i]
                p_comps.append((frame_i, cur))
            else:
                assert isinstance(p_comps, list), p_comps
                assert all(
                    isinstance(con, tuple) and
                    len(con) == 2 and
                    isinstance(con[1], set)
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
            print(f'\r{frame_i} - Warning: many to one')

        for i, con in enumerate(comps):
            if i not in cur_connected:
                prev_comps.append([(frame_i, con)])

        # fgmask = np.stack([fgmask for _ in range(3)], axis=2)
        if args.show and (args.shows is None or frame_i > args.shows[0]):
            frame = cv2.hconcat([frame, fgmask])
            cv2.imshow('window-name', frame)
            if args.fps is not None:
                time.sleep(1/args.fps)


    total_frame = frame_i+1
    if args.reverse:
        components = comp_reverse(components, total_frame)

    components = comp_sort(components)

    with open(args.pkl, 'wb') as f:
        pickle.dump(components, f)
    print(f'\nFound {len(components)} events, save to {args.pkl}')

    cv2.destroyAllWindows()  # destroy all opened windows
