import network
import cv2
import util_func
import numpy as np
import tensorflow as tf

net1 = network.load_network_1('net1_weights.h5')
net2 = network.load_network_2('net2_weights.h5')
net3 = network.load_network_3('net3_weights.h5')

graph = tf.get_default_graph()


def get_bounding_box(img, min_size=network.window_size, max_size=None, scale_factor=1.25, thresholds=[0.4, 0, 1], overlap_threshold=0.3):
    ret_bb = []
    ret_score = []
    rows, cols = img.shape[:2]
    if max_size is None:
        max_size = min(rows, cols)
    max_size = min(max_size, min(rows, cols))
    min_scale = float(network.window_size) / max_size
    max_scale = float(network.window_size) / min_size
    img = (img - 127.5) * 0.0078125
    scale = max_scale
    new_img = img
    while scale >= min_scale:
        new_rows = int(rows * scale)
        new_cols = int(cols * scale)
        if new_rows >= network.window_size and new_cols >= network.window_size:
            new_img = cv2.resize(new_img, (new_cols, new_rows), interpolation=cv2.INTER_AREA)

            # net1
            with graph.as_default():
                feature_map = net1.predict(np.array([new_img]))[0]
            bb, score = util_func.heatmap_to_bounding_box(feature_map, network.window_size, network.step_size, thresholds[0])

            # net2
            windows = []
            for br, bc, bs in bb:
                windows.append(new_img[br:br+bs, bc:bc+bs])
            if len(windows) > 0:
                windows = np.array(windows)
                with graph.as_default():
                    score = net2.predict(windows)
                keep_bb = []
                keep_score = []
                keep_windows = []
                for i in xrange(len(bb)):
                    if score[i, 0, 0, 0] > thresholds[1]:
                        keep_bb.append(bb[i])
                        keep_score.append(score[i])
                        keep_windows.append(windows[i])
                bb = keep_bb
                score = keep_score
                windows = keep_windows

            # net3
            if len(windows) > 0:
                windows = np.array(windows)
                with graph.as_default():
                    score = net3.predict(windows)
                keep_bb = []
                keep_score = []
                for i in xrange(len(bb)):
                    if score[i, 0, 0, 0] > thresholds[2]:
                        keep_bb.append(bb[i])
                        keep_score.append(score[i])
                bb = keep_bb
                score = keep_score

            # resize bounding box
            resized_bounding_box = [(int(r / scale), int(c / scale), int(s / scale)) for r, c, s in bb]
            ret_bb.extend(resized_bounding_box)
            ret_score.extend(score)
        scale /= scale_factor
    return util_func.non_max_suppress(ret_bb, ret_score, overlap_threshold)
