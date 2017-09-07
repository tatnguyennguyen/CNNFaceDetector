
def iou(r1, c1, s1, r2, c2, s2):
    l = max(c1, c2)
    r = min(c1 + s1, c2 + s2)
    t = max(r1, r2)
    b = min(r1 + s1, r2 + s2)
    if l > r or t > b:
        return 0
    s_i = (r - l) * (b - t)
    s_u = s1 * s1 + s2 * s2 - s_i
    return float(s_i) / s_u


def heatmap_to_bounding_box(fmap, window_size, step_size, threshold=0):
    rows, cols = fmap.shape[:2]
    bb = []
    score = []
    for i in xrange(rows):
        for j in xrange(cols):
            if fmap[i, j, 0] > threshold:
                r_in_img = i * step_size
                c_in_img = j * step_size
                bb.append((r_in_img, c_in_img, window_size))
                score.append(fmap[i, j, 0])
    return bb, score


def non_max_suppress(bb, score, overlap_threshold=0.5):
    ret_score = []
    ret_bb = []
    if len(bb) > 0:
        score, bb = zip(*sorted(zip(score, bb)))
        for i in xrange(len(bb)):
            for j in xrange(i+1, len(bb)):
                if iou(bb[i][0], bb[i][1], bb[i][2], bb[j][0], bb[j][1], bb[j][2]) > overlap_threshold:
                    break
            else:
                ret_bb.append(bb[i])
                ret_score.append(score[i])
    return ret_bb, ret_score
