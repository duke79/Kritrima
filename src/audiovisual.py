import logging
from threading import Thread
from time import time

import cv2
import mss
import numpy
import os
from matplotlib import pyplot
import matplotlib.animation as animation
import pylab

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)  # 'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'
log = logging.getLogger(__name__)

output_dir = os.path.abspath("output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class UnusedForNow:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def save_video2():
        dpi = 100
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

    @staticmethod
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

    @staticmethod
    def save_video_with_opencv_2():
        import numpy as np
        import cv2

        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'DIVX')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        # out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

        for idx, image in UnusedForNow.capture_screen():
            frame = cv2.flip(image, 0)

            # write the flipped frame
            out.write(frame)

            if idx > 50:
                break

        out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def save_screen():
        def save_one_screen_shot(image, name="screen"):
            UnusedForNow.save_picture(image, save_prefix=name)

        start_time = time()
        print(start_time)
        save_threads = []
        for idx, image in UnusedForNow.capture_screen():
            t = Thread(target=save_one_screen_shot, args=(image, "screen" + str(idx)))
            t.start()
            save_threads.append(t)
            if time() - start_time > 2:
                print(time())
                for t in save_threads:
                    t.join()
                exit()


class AudioVisual:
    def save_video_from_yt(self):
        import youtube_dl

        ydl_opts = {
            'outtmpl': output_dir + '/%(title)s.%(ext)s'
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])

    def load_video_frames_from_disk(self, path):
        log.info("Loading video frames for : \"" + os.path.basename(path) + "\"")
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        while success:
            yield count, image
            success, image = vidcap.read()
            count += 1

    def extract_audio_from_video(self, vid_path, aud_path="audio.wav"):
        log.info("Extracting audio from {0} into {1}".format(vid_path, aud_path))
        if os.path.exists(aud_path):
            os.remove(aud_path)

        import subprocess
        command = "ffmpeg -i \"{0}\" -ab 160k -acodec pcm_s16le -ac 2 -ar 44100 -vn \"{1}\"".format(vid_path, aud_path)
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # wait for the process to terminate
        for line in process.stdout:
            log.debug(line)
        for line in process.stderr:
            log.debug(line)
        errcode = process.returncode
        return aud_path

    def save_picture(self, path, image):
        cv2.imwrite(path, image)

    def save_video_frames(self, vid_path):
        for index, img in av.load_video_frames_from_disk(vid_path):
            vid_frames_dir = os.path.join(output_dir, "frames")
            if not os.path.exists(vid_frames_dir):
                os.makedirs(vid_frames_dir)
            output_file = os.path.join(vid_frames_dir, os.path.splitext(file)[0] + "_frame_" + str(index) + ".jpeg")
            av.save_picture(output_file, img)  # save frame as JPEG file


if __name__ == "__main__":
    av = AudioVisual()
    # av.save_video_from_yt()
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        # av.save_video_frames(file_path)
        av.extract_audio_from_video(file_path, os.path.join(output_dir, file + ".wav"))
