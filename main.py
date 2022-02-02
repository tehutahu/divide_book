import os
import sys

import cv2
import numpy as np


def save_page(img1, img2, page: int, out_dir: str) -> int:
    h, w = img1.shape[:2]
    size = (w // 4, h // 4)
    img1_, img2_ = [cv2.resize(img, size) for img in [img1, img2]]
    if np.allclose(img1_, img2_, rtol=0, atol=50):
        next_page = page
    else:
        result = cv2.imwrite(os.path.join(out_dir, str(page) + ".png"), img2)
        next_page = page + 1
    return next_page


def video_book2pages(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit()
    ret, frame_buff = cap.read()

    page = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("img", frame_buff)
        page = save_page(frame_buff, frame, page, out_dir)
        frame_buff = frame
        if cv2.waitKey(30) == 27:  # press esc
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    video_path = sys.argv[1]
    out_root = "out"
    out_dir = os.path.join(out_root, video_path)
    os.makedirs(out_dir, exist_ok=True)
    video_book2pages(video_path, out_dir)
