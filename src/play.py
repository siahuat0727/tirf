import cv2
import numpy as np
import json
import time

cap = cv2.VideoCapture("data/WT9.avi")
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

x1 = 50
x2 = 150
y1 = 50
y2 = 200


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
        if (ret:=dfs(xy))
    ]


def process(cons):
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
    }
    for k in info.keys():
        if k.startswith('x'):
            info[k] += x1
        if k.startswith('y'):
            info[k] += y1
    return info


def main():
    count = 0
    components = []
    prev_connects = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        print(f'\r{count} ', end='')
        frame = frame[x1:x2, y1:y2, 0]
        fgmask = fgbg.apply(frame)

        # if count < 1600:
        #     continue

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        connects = find_connected_component(fgmask, count)

        prev_cur = [
            [
                i
                for i, con in enumerate(connects)
                if con & prev_cons[-1][1]
            ]
            for prev_cons in prev_connects
        ]

        for curs, cons in zip(prev_cur, prev_connects):
            if curs:
                # assert len(curs) == 1, len(curs)
                cur = connects[curs[0]]
                for cur_i in curs[1:]:  # If any
                    print('Warning: one to many')
                    cur |= connects[cur_i]
                cons.append((count, cur))
            else:
                assert isinstance(cons, list), cons
                assert all(
                    isinstance(con, tuple) and len(con) == 2
                    for con in cons
                ), cons
                components.append(process(cons))
                print(sum(len(con) for _, con in cons))

        prev_connects = [
            con
            for curs, con in zip(prev_cur, prev_connects)
            if curs
        ]

        cur_connected = set(
            i
            for ll in prev_cur
            for i in ll
        )

        if len(cur_connected) != len(
                list(i for ll in prev_cur for i in ll)):
            print('Warning: many to one')

        for i, con in enumerate(connects):
            if i not in cur_connected:
                prev_connects.append([(count, con)])

        # fgmask = np.stack([fgmask for _ in range(3)], axis=2)
        # frame = cv2.hconcat([frame, fgmask])
        # cv2.imshow('window-name', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    with open('crop_list.txt', 'w') as f:
        print(components)
        json.dump(components, f)

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows


main()