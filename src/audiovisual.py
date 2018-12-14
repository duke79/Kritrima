from threading import Thread
from time import time

import cv2
import mss
import numpy
import os
from matplotlib import pyplot
import matplotlib.animation as animation
import pylab

output_dir = os.path.abspath("output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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


dpi = 100


def save_video2():
    fig = pylab.plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(pylab.rand(300, 300), cmap='gray', interpolation='nearest')
    im.set_clim([0, 1])
    fig.set_size_inches([5, 5])

    pylab.tight_layout()

    def update_img(n):
        tmp = pylab.rand(300, 300)
        im.set_data(tmp)
        return im

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, 300, interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('demo.mp4', writer=writer, dpi=dpi)
    return ani


def save_video_with_opencv():
    # For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    import cv2
    import os

    FILE_OUTPUT = 'output.avi'

    # Checks and deletes the output file
    # You cant have a existing file or it will through an error
    if os.path.isfile(FILE_OUTPUT):
        os.remove(FILE_OUTPUT)

    # Playing video from file:
    # cap = cv2.VideoCapture('vtest.avi')
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)

    currentFrame = 0

    # Get current width of frame
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'X264')
    out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20.0, (int(width), int(height)))

    # while(True):
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Handles the mirroring of the current frame
            frame = cv2.flip(frame, 1)

            # Saves for video
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Potential Error:
    # OpenCV: Cannot Use FaceTime HD Kamera
    # OpenCV: camera failed to properly initialize!
    # Segmentation fault: 11
    #
    # Solution:
    # I solved this by restarting my computer.
    # http://stackoverflow.com/questions/40719136/opencv-cannot-use-facetime/42678644#42678644


def save_video_with_opencv_2():
    import numpy as np
    import cv2

    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

    for idx, image in capture_screen():
        frame = cv2.flip(image, 0)

        # write the flipped frame
        out.write(frame)

        if idx > 50:
            break

    out.release()
    cv2.destroyAllWindows()


def save_screen():
    def save_one_screen_shot(image, name="screen"):
        save_picture(image, save_prefix=name)

    start_time = time()
    print(start_time)
    save_threads = []
    for idx, image in capture_screen():
        t = Thread(target=save_one_screen_shot, args=(image, "screen" + str(idx)))
        t.start()
        save_threads.append(t)
        if time() - start_time > 2:
            print(time())
            for t in save_threads:
                t.join()
            exit()


def save_video_from_yt():
    # from __future__ import unicode_literals
    import youtube_dl

    ydl_opts = {
        'outtmpl': output_dir + '/%(title)s.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])


# save_video()
# save_video2()
# save_video_with_opencv()
# save_video_with_opencv_2()
save_video_from_yt()
