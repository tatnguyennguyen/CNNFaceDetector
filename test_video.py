#!/usr/bin/env python
import cv2
import detect
import sys
import time

if __name__ == '__main__':
    if len(sys.argv) == 1:
        video_uri = 0
    else:
        video_uri = sys.argv[1]

    video = cv2.VideoCapture(video_uri)
    if not video.isOpened():
        print 'Cannot open video'
        exit(1)

    count_frame = 0
    begin = time.time()
    while True:
        ret, frame = video.read()
        if not ret:
            break

        count_frame += 1
        bb, score = detect.get_bounding_box(frame, 80, None, 1.4, [0.5, 1, 1])

        for r, c, s in bb:
            cv2.rectangle(frame, (c, r), (c+s, r+s), (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    end = time.time()
    print 'Frame rate : ', count_frame / (end - begin)
