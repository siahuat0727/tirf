import cv2
import numpy as np
import pickle
import time
import cc3d

from .utils import select_reader


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


def filter_prev_comps(prev_comps, frame_comps, frame_i, args):
    components = []

    # Find connection of prev_comps to frame_comps
    prev_cur = [
        [
            i
            for i, con in enumerate(frame_comps)
            if con & prev_cons[-1][1]
        ]
        for prev_cons in prev_comps
    ]

    cur_comps = []
    for curs, p_comps in zip(prev_cur, prev_comps):
        if curs:  # If prev connect to some frame_comps
            cur = frame_comps[curs[0]]
            for cur_i in curs[1:]:
                print(f'\n{frame_i} - Warning: one to many')
                cur |= frame_comps[cur_i]
            p_comps.append((frame_i, cur))
            cur_comps.append(p_comps)
        else:
            assert isinstance(p_comps, list), p_comps
            assert all(
                isinstance(con, tuple) and
                len(con) == 2 and
                isinstance(con[1], set)
                for con in p_comps
            ), p_comps
            components.append(process(p_comps, args))


    cur_connected = set(
        i
        for ll in prev_cur
        for i in ll
    )

    if len(cur_connected) != len(
            list(i for ll in prev_cur for i in ll)):
        print(f'\r{frame_i} - Warning: many to one')

    for i, con in enumerate(frame_comps):
        if i not in cur_connected:
            cur_comps.append([(frame_i, con)])
    return cur_comps, components


def process(cons, args):
    print(cons)
    zs, xs, ys = cons
    total_point = len(zs)
    info = {
        'frame_start': min(zs),
        'frame_end': max(zs),
        'frame': sum(zs) // total_point,
        'x_min': min(xs),
        'x_max': max(xs),
        'x': sum(xs) // total_point,
        'y_min': min(ys),
        'y_max': max(ys),
        'y': sum(ys) // total_point,
        'total_point': total_point,
        'regions': cons,
    }
    if args.x is not None:
        for k in info.keys():
            if k in ('x', 'x_min', 'x_max'):
                info[k] += args.x[0]
    if args.y is not None:
        for k in info.keys():
            if k in ('y', 'y_min', 'y_max'):
                info[k] += args.y[0]
    return info



def extract_comonents(cc, args):
    values, counts = np.unique(cc, return_counts=True)
    # sort and Ignore background
    values = list(sorted(values, key=counts.__getitem__, reverse=True))[1:]
    return [
        process(np.array((cc==v).nonzero()), args)
        for v in values
    ]


def play(args):

    # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    reader = select_reader(args.input_type)
    generator = reader(args.input, reverse=args.reverse)


    detect_imgs = np.array([
        cv2.morphologyEx(fgbg.apply(frame), cv2.MORPH_OPEN, kernel)
        for frame_i, frame in enumerate(generator)
    ])

    cc = cc3d.connected_components(detect_imgs, connectivity=26, delta=0)

    if args.reverse:
        cc = cc[::-1]

    components = extract_comonents(cc, args)

    components = comp_sort(components)

    with open(args.pkl, 'wb') as f:
        pickle.dump(components, f)
    print(f'\nFound {len(components)} events, save to {args.pkl}')
