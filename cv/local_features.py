from typing import Optional, Union
from argparse import ArgumentParser
from enum import Enum
import sys

import cv2 as cv
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow


colors = np.random.randint(0, 256, size=(1024, 3), dtype=np.uint8)

class Matcher:
    def __init__(self, alg: str):
        self.alg = alg
        # FLANN_INDEX_KDTREE = 1
        # index_params: dict[str, Union[bool, int, float, str]] = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params: dict[str, Union[bool, int, float, str]] = dict(checks=50)
        # self.matcher = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        # self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        if alg == "hog":
            pass
        elif alg == "harris":
            pass
        elif alg == "sift":
            FLANN_INDEX_KDTREE = 1
            index_params: dict[str, Union[bool, int, float, str]] = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params: dict[str, Union[bool, int, float, str]] = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

            sift = cv.SIFT_create()
            self.detect_and_compute = sift.detectAndCompute
        elif alg == "surf":
            # Non free
            pass
        elif alg == "orb":
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,      # 12 is good, 6 is faster
                                key_size=12,         # was 20 in many examples
                                multi_probe_level=1) # 1–2 is typical
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)

            orb = cv.ORB_create(nfeatures=512, scaleFactor=1.2, nlevels=8)
            self.detect_and_compute = orb.detectAndCompute


    def match_lines(self, im1, im2, roi):
        im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(im1)
        mask[roi[1]: roi[3], roi[0]:roi[2]] = True

        kp1, desc1 = self.detect_and_compute(im1, mask)
        kp2, desc2 = self.detect_and_compute(im2, None)

        # if self.alg == "sift":
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        keep_matches = []
        matches = list(filter(lambda elem: len(elem) == 2, matches))
        for nn, snn in matches:
            assert nn.queryIdx == snn.queryIdx
            if nn.distance < 0.7 * snn.distance:
                keep_matches.append(nn)
        # elif self.alg == "orb":
        #     matches = self.bf.match(desc1, desc2)
        #     print("matches: ", matches)
        #     matches = sorted(matches, key=lambda x: x.distance)
        #     keep_matches = matches


        lines = []
        for match in keep_matches:
            template_kp = kp1[match.queryIdx]
            matched_kp = kp2[match.trainIdx]
            matched_x, matched_y = matched_kp.pt
            matched_x += im1.shape[-1]
            line = (int(template_kp.pt[0]), int(template_kp.pt[1]), int(matched_x), int(matched_y))
            lines.append(line)
        return lines





class State(Enum):
    recording1 = 0
    snapping = 1
    recording2 = 2


class VideoWidget(QLabel):
    """
    QLabel that:
    - displays the 1x2 video grid (as a QPixmap)
    - handles mouse events to define the snapping rectangle
    """

    def __init__(self, app_logic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_logic = app_logic
        self.setMouseTracking(True)  # receive move events even without button pressed
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.app_logic.on_mouse_down(event.x(), event.y())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.app_logic.on_mouse_move(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.app_logic.on_mouse_up(event.x(), event.y())


class AppLogic:
    """
    Holds the OpenCV/video logic and drawing state.
    """

    def __init__(self, args, video_widget: VideoWidget):
        self.args = args
        self.video_widget = video_widget
        self.matcher = Matcher(alg=args.alg_name)

        self.state = State.recording1

        # Rectangle selection state
        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.curr_x: Optional[int] = None
        self.curr_y: Optional[int] = None

        # Video / canvas
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("failed to start webcam")

        # Try to set size
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("failed to read from camera")
        self.height, self.width = frame.shape[:2]

        self.snapshot = None
        self.canvas = np.zeros((self.height, self.width * 2, 3), dtype=np.uint8)

        # Keep reference to the last RGB frame so QImage has valid memory
        self._last_rgb = None

    # ---------- Mouse logic (called from VideoWidget) ----------

    def _clamp_to_canvas(self, x, y):
        # Clamp mouse coords to the canvas area in the left cell
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        return x, y


    def on_mouse_down(self, x, y):
        if self.state in (State.recording2, State.snapping):
            return

        x, y = self._clamp_to_canvas(x, y)
        self.state = State.snapping
        self.start_x, self.start_y = x, y
        self.curr_x, self.curr_y = x, y


    def on_mouse_move(self, x, y):
        if self.state != State.snapping:
            return
        self.curr_x, self.curr_y = self._clamp_to_canvas(x, y)


    def on_mouse_up(self, x, y):
        if self.state != State.snapping:
            return
        self.curr_x, self.curr_y = self._clamp_to_canvas(x, y)
        self.state = State.recording2


    # ---------- Main update loop (called from QTimer) ----------

    def update(self):
        # Always grab a new frame
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.args.flip:
            frame = cv.flip(frame, flipCode=1)

        # Reset canvas
        self.canvas[:] = 0

        if self.state in (State.recording1, State.snapping):
            # Left cell: live video
            self.canvas[:self.height, :self.width, :] = frame

            # While snapping, keep updating snapshot (we'll freeze it on mouse up)
            if self.state == State.recording1:
                self.snapshot = frame
            elif self.state == State.snapping:
                self.snapshot = self.snapshot.copy()

            # Instruction text
            cv.putText(
                self.canvas,
                "Click and move mouse to select region of interest",
                (10, self.height - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )

        elif self.state == State.recording2:
            self.canvas[:self.height, :self.width, :] = self.snapshot
            # Right cell: live video
            self.canvas[:self.height, self.width:, :] = frame

        # Draw rectangle if we have a selection
        if self.start_x is not None and self.curr_x is not None:
            min_x = min(self.start_x, self.curr_x)
            min_y = min(self.start_y, self.curr_y)
            max_x = max(self.start_x, self.curr_x)
            max_y = max(self.start_y, self.curr_y)

            rect_min_x = max(min_x, 0)
            rect_min_y = max(min_y, 0)
            rect_max_x = min(max_x, self.width - 1)
            rect_max_y = min(max_y, self.height - 1)

            cv.rectangle(
                self.canvas,
                pt1=(rect_min_x, rect_min_y),
                pt2=(rect_max_x, rect_max_y),
                color=(255, 255, 255),
                thickness=1,
                lineType=cv.LINE_AA,
            )

            if self.state == State.recording2:
                matched_lines = self.matcher.match_lines(self.snapshot, frame, roi=(rect_min_x, rect_min_y, rect_max_x, rect_max_y))
                for i, line in enumerate(matched_lines):
                    cv.line(
                        self.canvas,
                        pt1=(line[0], line[1]),
                        pt2=(line[2], line[3]),
                        color=colors[i].tolist(),
                        thickness=1,
                        lineType=cv.LINE_AA
                    )


        # Convert canvas (BGR) → QImage (RGB)
        rgb = cv.cvtColor(self.canvas, cv.COLOR_BGR2RGB)
        self._last_rgb = rgb  # keep ref so QImage memory stays valid
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        self.video_widget.setPixmap(pixmap)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.setWindowTitle("1x2 Video Grid (PyQt5)")

        self.video_widget = VideoWidget(None)
        self.setCentralWidget(self.video_widget)

        self.app_logic = AppLogic(args, self.video_widget)
        self.video_widget.app_logic = self.app_logic  # circular link

        w = self.app_logic.width * 2
        h = self.app_logic.height
        self.video_widget.setFixedSize(w, h)
        self.setFixedSize(w, h)

        # Timer: ~30 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.app_logic.update)
        self.timer.start(33)  # ms

        # Resize window to fit content better
        self.resize(self.app_logic.width * 2, self.app_logic.height)

    def closeEvent(self, event):
        # Clean up camera when window closes
        self.app_logic.release()
        event.accept()


def main(args):
    app = QApplication(sys.argv)
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["hog", "harris", "sift", "surf", "orb"],
        default="sift",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
    )
    parser.add_argument(
        "--matcher",
        choices=["bf", "flann"],
        default="bf"
    )

    args = parser.parse_args()
    main(args)