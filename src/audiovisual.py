from threading import Thread
from time import time

import cv2
import mss
import numpy
import os
from matplotlib import pyplot


def capture_screen():
    with mss.mss() as sct:
        # Part of the screen to capture
        # monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
        monitor = sct.monitors[1]

        counter = 0
        while "Screen capturing":
            counter = counter + 1
            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))
            yield counter, img


def save_picture(img, grayscale=False, display=False, save=True, save_prefix="screen"):
    if display:
        if grayscale:
            pyplot.imshow(img, cmap="gray")
            # cv2.imshow('OpenCV/Numpy grayscale',
            #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
        else:
            cv2.imshow("OpenCV/Numpy normal", img)
    if save:
        output_dir = os.path.abspath("output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # pyplot.savefig("screen.png")
        import scipy.misc
        # scipy.misc.toimage(img, cmin=0.0, cmax=...).save('outfile.jpg')
        scipy.misc.imsave(os.path.join(output_dir, '{0}.png'.format(save_prefix)), img)


def save_video():
    image_folder = 'images'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def save_screen_shot(image, name="screen"):
    save_picture(image, save_prefix=name)


start_time = time()
print(start_time)
save_threads = []
for idx, image in capture_screen():
    t = Thread(target=save_screen_shot, args=(image, "screen" + str(idx)))
    t.start()
    save_threads.append(t)
    if time() - start_time > 2:
        print(time())
        for t in save_threads:
            t.join()
        exit()
