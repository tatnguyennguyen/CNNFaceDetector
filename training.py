#!/usr/bin/env python
import network
import cv2
import random
import sys
import numpy as np
import keras
import keras.backend as K
import util_func
import tensorflow as tf


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


graph = None

window_size = network.window_size
step_size = network.step_size

stage_1_threshold = 0
stage_2_threshold = 0

max_neg_window_per_image = 4

max_processed_count = 100

n_can_overlap = (window_size / 3 / step_size) ** 2


def non_overlap(bb, max_window=float('inf'), threshold=0.5):
    ret_bb = []
    if len(bb) > 0:
        for i in xrange(len(bb)):
            for j in xrange(i+1, len(bb)):
                if util_func.iou(bb[i][0], bb[i][1], bb[i][2], bb[j][0], bb[j][1], bb[j][2]) > threshold:
                    break
            else:
                ret_bb.append(bb[i])
                if len(ret_bb) >= max_window:
                    break
    return ret_bb


class DataGen_1():
    def __init__(self, txt_file, net1, net2, batch_size=64):
        self.batch_size = batch_size
        self.net1 = net1
        self.net2 = net2
        self.items = []
        self.stage = 0  # 0: random crop; 1: use net1 for mining example; 2: use net1 and net2 for mining example
        with open(txt_file) as fi:
            lines = fi.readlines()
            i = 0
            while i < len(lines):
                fn = lines[i]
                i += 1
                item = dict()
                item['filename'] = fn.strip()
                n_face = int(lines[i])
                i += 1
                faces = []
                accept = True
                for j in range(n_face):
                    r, c, h, w = [int(float(f)) for f in lines[i].split()]
                    if w != h:
                        print fn, ': Accept only square annotation'
                        accept = False
                    i += 1
                    s = w
                    faces.append((r, c, s))
                item['faces'] = faces
                if accept:
                    self.items.append(item)

    def __iter__(self):
        return self

    def next(self):
        ret_image = []
        ret_label = []
        n_face_need = self.batch_size / 2
        n_bg_need = self.batch_size - n_face_need

        # get face windows
        while n_face_need > 0:
            item = np.random.choice(self.items)
            img = cv2.imread(item['filename'])
            if img is None:
                print 'Cannot read ', item['filename']
                continue
            img = (img - 127.5) * 0.0078125
            rows, cols = img.shape[:2]
            for r, c, s in item['faces']:
                new_s = (random.randint(int(s / 1.1892), int(s * 1.1892)) + random.randint(int(s / 1.1892), int(s * 1.1892))) / 2  # https://en.wikipedia.org/wiki/Bates_distribution
                new_s = min(new_s, rows, cols)
                new_r = r - (new_s - s) / 2  # align
                new_c = c - (new_s - s) / 2  # align
                step = int(new_s * (float(step_size) / window_size))
                new_r += (random.randint(-step / 2, step / 2) + random.randint(-step / 2, step / 2)) / 2
                new_c += (random.randint(-step / 2, step / 2) + random.randint(-step / 2, step / 2)) / 2
                new_r = max(0, new_r)
                new_c = max(0, new_c)
                new_r_e = new_r + new_s
                new_c_e = new_c + new_s
                if new_r_e > rows or new_c_e > cols or util_func.iou(r, c, s, new_r, new_c, new_s) < 0.5:
                    continue
                face = img[new_r:new_r_e, new_c:new_c_e]
                face = cv2.resize(face, (window_size, window_size), interpolation=cv2.INTER_AREA)
                if random.random() < 0.5:
                    face = cv2.flip(face, 1)
                ret_image.append(face)
                ret_label.append(1)
                n_face_need -= 1
                if n_face_need <= 0:
                    break

        # get background windows
        processed_count = 0
        while n_bg_need > 0:
            item = np.random.choice(self.items)
            img = cv2.imread(item['filename'])
            if img is None:
                print 'Cannot read ', item['filename']
                continue
            processed_count += 1
            img = (img - 127.5) * 0.0078125
            rows, cols = img.shape[:2]
            min_size = random.randint(window_size, int(1.4 * window_size))
            max_size = min(rows, cols)
            min_scale = float(window_size) / max_size
            max_scale = float(window_size) / min_size
            scales = []
            s = min_scale
            while s <= max_scale:
                scales.append(s)
                s *= 1.4

            random.shuffle(scales)

            this_image_bg_count = 0

            for scale in scales:
                new_rows = int(rows * scale)
                new_cols = int(cols * scale)
                if new_rows < window_size or new_cols < window_size:
                    continue
                new_img = cv2.resize(img, (new_cols, new_rows), interpolation=cv2.INTER_AREA)

                if self.stage == 0:
                    bb = []
                    for i in xrange(0, new_rows - window_size + 1, step_size):
                        for j in xrange(0, new_cols - window_size + 1, step_size):
                            bb.append((i, j, window_size))
                            if len(bb) > (max_neg_window_per_image + len(item['faces'])) * n_can_overlap:
                                break

                if self.stage >= 1:
                    with graph.as_default():
                        fmap = self.net1.predict(np.array([new_img]))[0]
                    bb, _ = util_func.heatmap_to_bounding_box(fmap, window_size, step_size, stage_1_threshold)

                if self.stage >= 2:
                    if len(bb) > 0:
                        windows = []
                        for br, bc, bs in bb:
                            windows.append(new_img[br:br+bs, bc:bc+bs])
                        windows = np.array(windows)
                        with graph.as_default():
                            score = self.net2.predict(windows)
                        keep_bb = []
                        for i in xrange(len(bb)):
                            if score[i, 0, 0, 0] > stage_2_threshold:
                                keep_bb.append(bb[i])
                        bb = keep_bb

                if len(bb) > 0:
                    random.shuffle(bb)
                    bb = non_overlap(bb, (max_neg_window_per_image + len(item['faces'])) * n_can_overlap)

                    for br, bc, bs in bb:
                        br_e = br + bs
                        bc_e = bc + bs
                        if br >= 0 and br_e <= new_rows and bc >= 0 and bc_e <= new_cols:
                            for r, c, s in item['faces']:
                                r = r * scale
                                c = c * scale
                                s = s * scale
                                # print len(ret_image), util_func.iou(r, c, s, br, bc, bs), len(item['faces']), item['filename']
                                if util_func.iou(r, c, s, br, bc, bs) > 0.35:
                                    break
                            else:  # breaking was not happend
                                ret_image.append(new_img[br:br_e, bc:bc_e])
                                ret_label.append(-1)
                                processed_count = 0
                                n_bg_need -= 1
                                this_image_bg_count += 1
                                if n_bg_need <= 0:
                                    break  # for br, bc, bs in bb:
                                if this_image_bg_count >= max_neg_window_per_image:
                                    break  # for br, bc, bs in bb:

                if n_bg_need <= 0:
                    break  # for scale in scales
                if this_image_bg_count >= max_neg_window_per_image:
                    break  # for scale in scales
            if processed_count > max_processed_count:
                raise StopIteration()

        ret_image = np.array(ret_image[:self.batch_size])
        ret_label = np.array(ret_label[:self.batch_size])
        ret_label = np.reshape(ret_label, (-1, 1, 1, 1))
        return ret_image, ret_label


class DataGen_2():
    def __init__(self, txt_file, bg_txt, net1, net2, batch_size=64):
        self.batch_size = batch_size
        self.net1 = net1
        self.net2 = net2
        self.items = []
        self.stage = 0  # 0: random crop; 1: use net1 for mining example; 2: use net1 and net2 for mining example
        with open(txt_file) as fi:
            lines = fi.readlines()
            i = 0
            while i < len(lines):
                fn = lines[i]
                i += 1
                item = dict()
                item['filename'] = fn.strip()
                n_face = int(lines[i])
                i += 1
                faces = []
                accept = True
                for j in range(n_face):
                    r, c, h, w = [int(float(f)) for f in lines[i].split()]
                    if w != h:
                        print fn, ': Accept only square annotation'
                        accept = False
                    i += 1
                    s = w
                    faces.append((r, c, s))
                item['faces'] = faces
                if accept:
                    self.items.append(item)

        self.bg_file = []
        with open(bg_txt) as fi:
            for line in fi:
                line = line.strip()
                if line:
                    self.bg_file.append(line)

    def __iter__(self):
        return self

    def next(self):
        ret_image = []
        ret_label = []
        n_face_need = self.batch_size / 2
        n_bg_need = self.batch_size - n_face_need

        # get face windows
        while n_face_need > 0:
            item = np.random.choice(self.items)
            img = cv2.imread(item['filename'])
            if img is None:
                print 'Cannot read ', item['filename']
                continue
            img = (img - 127.5) * 0.0078125
            rows, cols = img.shape[:2]
            for r, c, s in item['faces']:
                new_s = (random.randint(int(s / 1.1892), int(s * 1.1892)) + random.randint(int(s / 1.1892), int(s * 1.1892))) / 2  # https://en.wikipedia.org/wiki/Bates_distribution
                new_s = min(new_s, rows, cols)
                new_r = r - (new_s - s) / 2  # align
                new_c = c - (new_s - s) / 2  # align
                step = int(new_s * (float(step_size) / window_size))
                new_r += (random.randint(-step / 2, step / 2) + random.randint(-step / 2, step / 2)) / 2
                new_c += (random.randint(-step / 2, step / 2) + random.randint(-step / 2, step / 2)) / 2
                new_r = max(0, new_r)
                new_c = max(0, new_c)
                new_r_e = new_r + new_s
                new_c_e = new_c + new_s
                if new_r_e > rows or new_c_e > cols or util_func.iou(r, c, s, new_r, new_c, new_s) < 0.5:
                    continue
                face = img[new_r:new_r_e, new_c:new_c_e]
                face = cv2.resize(face, (window_size, window_size), interpolation=cv2.INTER_AREA)
                if random.random() < 0.5:
                    face = cv2.flip(face, 1)
                ret_image.append(face)
                ret_label.append(1)
                n_face_need -= 1
                if n_face_need <= 0:
                    break

        # get background windows
        processed_count = 0
        while n_bg_need > 0:
            filename = np.random.choice(self.bg_file)
            img = cv2.imread(filename)
            if img is None:
                print 'Cannot read ', filename
                continue
            processed_count += 1
            img = (img - 127.5) * 0.0078125
            rows, cols = img.shape[:2]
            min_size = random.randint(window_size, int(1.4 * window_size))
            max_size = min(rows, cols)
            min_scale = float(window_size) / max_size
            max_scale = float(window_size) / min_size
            scales = []
            s = min_scale
            while s <= max_scale:
                scales.append(s)
                s *= 1.4

            random.shuffle(scales)

            this_image_bg_count = 0

            for scale in scales:
                new_rows = int(rows * scale)
                new_cols = int(cols * scale)
                if new_rows < window_size or new_cols < window_size:
                    continue
                new_img = cv2.resize(img, (new_cols, new_rows), interpolation=cv2.INTER_AREA)

                if self.stage == 0:
                    bb = []
                    for i in xrange(0, new_rows - window_size + 1, step_size):
                        for j in xrange(0, new_cols - window_size + 1, step_size):
                            bb.append((i, j, window_size))
                            if len(bb) > (max_neg_window_per_image) * n_can_overlap:
                                break

                if self.stage >= 1:
                    with graph.as_default():
                        fmap = self.net1.predict(np.array([new_img]))[0]
                    bb, _ = util_func.heatmap_to_bounding_box(fmap, window_size, step_size, stage_1_threshold)

                if self.stage >= 2:
                    if len(bb) > 0:
                        windows = []
                        for br, bc, bs in bb:
                            windows.append(new_img[br:br+bs, bc:bc+bs])
                        windows = np.array(windows)
                        with graph.as_default():
                            score = self.net2.predict(windows)
                        keep_bb = []
                        for i in xrange(len(bb)):
                            if score[i, 0, 0, 0] > stage_2_threshold:
                                keep_bb.append(bb[i])
                        bb = keep_bb

                if len(bb) > 0:
                    random.shuffle(bb)
                    bb = non_overlap(bb, (max_neg_window_per_image) * n_can_overlap)

                    for br, bc, bs in bb:
                        br_e = br + bs
                        bc_e = bc + bs
                        if br >= 0 and br_e <= new_rows and bc >= 0 and bc_e <= new_cols:
                            ret_image.append(new_img[br:br_e, bc:bc_e])
                            ret_label.append(-1)
                            processed_count = 0
                            n_bg_need -= 1
                            this_image_bg_count += 1
                            if n_bg_need <= 0:
                                break  # for br, bc, bs in bb:
                            if this_image_bg_count >= max_neg_window_per_image:
                                break  # for br, bc, bs in bb:

                if n_bg_need <= 0:
                    break  # for scale in scales
                if this_image_bg_count >= max_neg_window_per_image:
                    break  # for scale in scales

            if processed_count > max_processed_count:
                raise StopIteration()

        ret_image = np.array(ret_image[:self.batch_size])
        ret_label = np.array(ret_label[:self.batch_size])
        ret_label = np.reshape(ret_label, (-1, 1, 1, 1))
        return ret_image, ret_label


class DataGen():
    def __init__(self, gen1, gen2):
        self.gen1 = gen1
        self.gen2 = gen2

    def __iter__(self):
        return self

    def next(self):
        if random.random() < 0.5:
            return self.gen1.next()
        else:
            return self.gen2.next()


def accu(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1)


def learn_rate_function(init_learn_rate):
    def learn_rate(epoch):
        return init_learn_rate * 1 / (1 + 0.01 * epoch)
    return learn_rate


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print 'Usage: python training.py txt_file [pretrain_weight_net1 [pretrain_weight_net2 [pretrain_weight_net3]]]'
        exit(0)

    txt_file = sys.argv[1]

    net1 = network.get_network_1()
    net2 = network.get_network_2()
    net3 = network.get_network_3()
    if len(sys.argv) >= 3:
        net1.load_weights(sys.argv[2])
    if len(sys.argv) >= 4:
        net2.load_weights(sys.argv[3])
    if len(sys.argv) == 5:
        net3.load_weights(sys.argv[4])
    net1.summary()
    net2.summary()
    net3.summary()

    global graph
    graph = tf.get_default_graph()

    batch_size = 32
    data_gen_1 = DataGen_1(txt_file, net1, net2, batch_size)
    data_gen_2 = DataGen_2(txt_file, 'list_background_file.txt', net1, net2, batch_size)
    data_gen = DataGen(data_gen_1, data_gen_2)
    # data_gen = data_gen_2

    # print 'Get windows for viewing'
    # data_gen.stage = 2
    # image, label = data_gen.next()
    # print len(image), len(label)
    # for i in xrange(len(image)):
    #     if label[i] == 1:
    #         cv2.imwrite('/tmp/face_' + str(i) + '.jpg', image[i] / 0.0078125 + 127.5)
    #     else:
    #         cv2.imwrite('/tmp/bg_' + str(i) + '.jpg', image[i] / 0.0078125 + 127.5)
    # exit()

    # Training net1
    print 'Training net1'
    data_gen_1.stage = 0
    data_gen_2.stage = 0
    print 'Getting validation data'
    val_image, val_label = data_gen.next()
    for i in xrange(2000 / batch_size):
        tmp_image, tmp_label = data_gen.next()
        val_image = np.append(val_image, tmp_image, axis=0)
        val_label = np.append(val_label, tmp_label, axis=0)
        sys.stdout.write('.')
        sys.stdout.flush()
    print ''
    saver = keras.callbacks.ModelCheckpoint('net1_weights_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, period=1)
    lr_scheduler = keras.callbacks.LearningRateScheduler(learn_rate_function(1e-5))
    opt = keras.optimizers.Adam()
    net1.compile(loss='hinge', optimizer=opt, metrics=[accu])
    try:
        net1.fit_generator(data_gen, steps_per_epoch=100, epochs=100, verbose=1, callbacks=[saver, lr_scheduler], validation_data=(val_image, val_label))
    except StopIteration:
        print 'Stop training. Not enough examples'
    net1.save_weights('net1_weights_end.h5')

    # Training net2
    print 'Training net2'
    data_gen_1.stage = 1
    data_gen_2.stage = 1
    print 'Getting validation data'
    val_image, val_label = data_gen.next()
    for i in xrange(2000 / batch_size):
        tmp_image, tmp_label = data_gen.next()
        val_image = np.append(val_image, tmp_image, axis=0)
        val_label = np.append(val_label, tmp_label, axis=0)
        sys.stdout.write('.')
        sys.stdout.flush()
    print ''
    saver = keras.callbacks.ModelCheckpoint('net2_weights_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, period=1)
    lr_scheduler = keras.callbacks.LearningRateScheduler(learn_rate_function(1e-5))
    opt = keras.optimizers.Adam()
    net2.compile(loss='hinge', optimizer=opt, metrics=[accu])
    try:
        net2.fit_generator(data_gen, steps_per_epoch=100, epochs=100, verbose=1, callbacks=[saver, lr_scheduler], validation_data=(val_image, val_label))
    except StopIteration:
        print 'Stop training. Not enough examples'
    net2.save_weights('net2_weights_end.h5')

    # Training net3
    print 'Training net3'
    data_gen_1.stage = 2
    data_gen_2.stage = 2
    print 'Getting validation data'
    val_image, val_label = data_gen.next()
    for i in xrange(2000 / batch_size):
        tmp_image, tmp_label = data_gen.next()
        val_image = np.append(val_image, tmp_image, axis=0)
        val_label = np.append(val_label, tmp_label, axis=0)
        sys.stdout.write('.')
        sys.stdout.flush()
    print ''
    saver = keras.callbacks.ModelCheckpoint('net3_weights_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, period=1)
    lr_scheduler = keras.callbacks.LearningRateScheduler(learn_rate_function(1e-5))
    opt = keras.optimizers.Adam()
    net3.compile(loss='hinge', optimizer=opt, metrics=[accu])
    try:
        net3.fit_generator(data_gen, steps_per_epoch=100, epochs=100, verbose=1, callbacks=[saver, lr_scheduler], validation_data=(val_image, val_label))
    except StopIteration:
        print 'Stop training. Not enough examples'
    net3.save_weights('net3_weights_end.h5')


if __name__ == '__main__':
    main()
