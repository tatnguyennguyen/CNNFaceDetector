#!/usr/bin/env python
import cv2
import detect
import sys
import time

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'Usage: ./test_image.py image1 image2 image3 ...'
        exit(0)

    begin = time.time()
    for fn in sys.argv[1:]:
        print fn
        img = cv2.imread(fn)
        if img is None:
            print 'Cannot read ', sys.argv[1]
            exit(1)

        rows, cols = img.shape[:2]
        new_img = cv2.resize(img, (cols, rows))

        bb, score = detect.get_bounding_box(new_img, 29, None, 1.25, [0.4, 0, 1], 0.3)
        print 'Detected', len(bb), 'faces'

        for r, c, s in bb:
            cv2.rectangle(img, (c, r), (c+s, r+s), (0, 255, 0), 2)

        cv2.imwrite(fn+'_out.png', img)
    end = time.time()
    print 'Times ', end - begin, ' (s)'
