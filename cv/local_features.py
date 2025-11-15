# from typing import Optional, List, Tuple
# from argparse import ArgumentParser
# from enum import Enum

# import cv2 as cv
# import numpy as np


# class State(Enum):
#     recording1 = 0
#     snapping = 1
#     recording2 = 2

# state = State.recording1
# min_x: Optional[int] = None
# min_y: Optional[int] = None
# max_x: Optional[int] = None
# max_y: Optional[int] = None
# lines: List[Tuple[int, ...]] = []


# def mouse_cb(event, x, y, flags, param):
#     global state, min_x, min_y, max_x, max_y

#     if state == State.recording2:
#         return

#     if event == cv.EVENT_LBUTTONDOWN:
#         state = State.snapping
#         min_x = max_x = x
#         min_y = max_y = y
#     elif event == cv.EVENT_MOUSEMOVE and state == State.snapping:

#         max_x = max(min_x, x)
#         max_y = max(min_y, y)
#     elif event == cv.EVENT_LBUTTONUP and state == State.snapping:
#         state = State.recording2


# def main(args):
#     global state, min_x, min_y, max_x, max_y
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         return print("failed to start webcam")
#     ret, frame = cap.read()
#     if not ret:
#         return print("failed to read from camera")
#     height, width = frame.shape[:2]

#     window_name = "1x2 Video Grid"
#     cv.namedWindow(window_name)
#     cv.setMouseCallback(window_name, mouse_cb)

#     snapshot = None
#     canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)
#     while True:
#         if state == State.recording1 or state == State.recording2:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = (
#                 cv.flip(frame, flipCode=1)
#                 if args.flip
#                 else frame
#             )
#         else:
#             snapshot = (
#                 cv.flip(frame, flipCode=1)
#                 if args.flip
#                 else frame
#             )


#         if state == State.recording1 or state == State.snapping:
#             canvas[:height, :width, :] = frame
#             cv.putText(
#                 canvas,
#                 "Click and move mouse to select region of interest",
#                 (10, height-10),
#                 cv.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 255, 255),
#                 1,
#                 cv.LINE_AA
#             )
#         elif state == State.recording2:
#             canvas[:height, :width, :] = snapshot
#             canvas[:height, width:, :] = frame

#         if min_x is not None:
#             rect_min_x = max(min_x, 0)
#             rect_min_y = max(min_y, 0)
#             rect_max_x = min(max_x, width - 1)
#             rect_max_y = min(max_y, height - 1)
#             canvas = cv.rectangle(
#                 canvas,
#                 pt1=(rect_min_x, rect_min_y),
#                 pt2=(rect_max_x, rect_max_y),
#                 color=(255, 255, 255),
#                 thickness=1
#             )

#         cv.imshow(window_name, canvas)
#         key = cv.waitKey(40) & 0xFF
#         if key == 27:
#             break

#     cap.release()
#     cv.destroyAllWindows()


# if __name__ == "__main__":
#     argparse = ArgumentParser()
#     argparse.add_argument(
#         "--alg_name",
#         choices=["hog", "harris", "sift", "surf", "orb"],
#         default="hog"
#     )
#     argparse.add_argument(
#         "--flip",
#         default=False
#     )

#     args = argparse.parse_args()
#     main(args)

from typing import Optional
from argparse import ArgumentParser
from enum import Enum
import sys

import cv2 as cv
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow


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
            # Left cell: snapshot from snapping
            if self.snapshot is None:
                self.snapshot = frame.copy()
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

            print(rect_min_x, rect_min_y)

            cv.rectangle(
                self.canvas,
                pt1=(rect_min_x, rect_min_y),
                pt2=(rect_max_x, rect_max_y),
                color=(255, 255, 255),
                thickness=1,
                lineType=cv.LINE_AA,
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
        default="hog",
    )
    parser.add_argument(
        "--flip",
        action="store_true",  # better than default=False
    )

    args = parser.parse_args()
    main(args)