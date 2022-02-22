from typing import Union
import cv2
import math as m
import numpy as np


def read_image_rgb(fpath: str) -> np.array:
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_image_rgb(fpath: str, img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fpath, img)


def toGray(img):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def extract_subimg_roi(img, xywh):
    x, y, w, h = xywh
    roi = img[y : y + h, x : x + w]

    return roi


def extract_perspective_roi(img, src_pnts, target_size=400):
    if isinstance(target_size, (int, float)):
        target_size = (target_size, target_size)

    src_pnts = np.array(src_pnts, dtype=np.float32)

    target_pnts = np.array(
        [
            [0, 0],
            [target_size[0], 0],
            [target_size[0], target_size[1]],
            [0, target_size[1]],
        ],
        dtype=np.float32,
    )

    perspective_mtrx = cv2.getPerspectiveTransform(src_pnts, target_pnts)
    transformed_img = cv2.warpPerspective(img, perspective_mtrx, target_size)

    return transformed_img, perspective_mtrx


def extract_screen_roi(img):
    thresh_img = binarizeOtsu(img)

    morph_kernel = np.ones((5, 5), np.uint8)
    morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel)

    return morph_img


class ArrowDetector:
    def __init__(self) -> None:
        pass

    def transform(self, screen_frame, threshold: int = 90):
        roi_thresh_img = binarizeOtsu(screen_frame, filter_sz=0)

        morph = cv2.morphologyEx(
            np.invert(roi_thresh_img),
            cv2.MORPH_OPEN,
            kernel=np.ones((5, 5), dtype=np.uint8),
        )
        cleaned_img = np.invert(roi_thresh_img) - morph

        hough_commot_opts = dict(
            image=cleaned_img,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            lines=None,
            srn=0,
            stn=0,
        )

        # Hough (0-45, 135-180)
        lines1 = cv2.HoughLines(
            min_theta=0,
            max_theta=np.pi / 4,
            **hough_commot_opts,
        )
        lines2 = cv2.HoughLines(
            min_theta=np.pi * 3 / 4,
            max_theta=np.pi,
            **hough_commot_opts,
        )

        lines = []

        if lines1 is not None:
            lines.append(lines1)

        if lines2 is not None:
            lines.append(lines2)

        if len(lines) > 0:
            lines = np.concatenate(lines)

        angle_values = []

        if len(lines) > 0:
            angle_values = np.array(lines)[:, 0, 1]
            np.mean(angle_values)
            np.std(angle_values)

        return lines, angle_values


def normalize_angles_deg(x: np.ndarray, copy: bool = True):
    if copy:
        x = x.copy()

    mask = x > 90
    x[mask] = x[mask] - 180
    return x


class IndicatorScreenExtractor:
    def __init__(
        self, roi_pnts: Union[list, np.ndarray], ind_size: Union[int, tuple] = 400
    ) -> None:
        self._ind_size = ind_size

        if isinstance(self._ind_size, (int, float)):
            self._ind_size = (self._ind_size, self._ind_size)

        self._obj_perspective_mtrx = self._get_initial_transform(
            roi_pnts, self._ind_size
        )

    @staticmethod
    def _get_initial_transform(
        src_pnts: Union[list, np.ndarray], target_size: Union[list, tuple]
    ):
        src_pnts = np.array(src_pnts, dtype=np.float32)

        target_pnts = np.array(
            [
                [0, 0],
                [target_size[0], 0],
                [target_size[0], target_size[1]],
                [0, target_size[1]],
            ],
            dtype=np.float32,
        )

        perspective_mtrx = cv2.getPerspectiveTransform(src_pnts, target_pnts)
        return perspective_mtrx

    def fit(self, img):
        transformed_img = cv2.warpPerspective(
            img, self._obj_perspective_mtrx, self._ind_size
        )

        morph_image = extract_screen_roi(transformed_img)

        self.screen_bbox_xywh = find_largest_contour(morph_image)

    def transform(self, img):
        transformed_img = cv2.warpPerspective(
            img, self._obj_perspective_mtrx, self._ind_size
        )

        screen_roi = extract_subimg_roi(transformed_img, self.screen_bbox_xywh)

        return screen_roi


class AngleUnitsConvertor:
    def __init__(self) -> None:
        self.model = None

    def fit(self, X: Union[list, np.ndarray], y: Union[list, np.ndarray]):
        z = np.polyfit(x=X, y=y, deg=1)
        self.model = np.poly1d(z)

    def transform(self, X):
        return self.model(X)


class ImageTextWriter:
    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, 500)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)
        self.thickness = 2

    def putText(self, img: np.ndarray, text: str, offset: tuple):
        # offset - top-left corner offset
        cv2.putText(
            img,
            text,
            offset,
            self.font,
            self.fontScale,
            self.fontColor,
            self.thickness,
        )


class VideoReader:
    def __init__(self, fpath) -> None:
        self._cap = cv2.VideoCapture(fpath)

    def read_frame(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __getitem__(self, index):
        self.move_2_frame(index)
        return self.read_frame()

    def move_2_frame(self, index):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    def __len__(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def size(self):
        # in CV format - WH
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )


def find_largest_contour(img):
    # morph_kernel = np.ones((5,5),np.uint8)
    # morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    bbox_xywh = cv2.boundingRect(cnt)

    return bbox_xywh


def adaptiveThreshold(img, filter_sz=3, block_sz=199, constant=5):
    img = toGray(img)

    if filter_sz > 0:
        blur = cv2.GaussianBlur(img, (filter_sz, filter_sz), 0)
    else:
        blur = img

    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_sz, constant
    )
    return th


def binarizeOtsu(img, filter_sz=3):
    img = toGray(img)

    if filter_sz > 0:
        blur = cv2.GaussianBlur(img, (filter_sz, filter_sz), 0)
    else:
        blur = img

    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def threshold(img, threshold=127):
    img = toGray(img)

    ret, th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return th


def filter_lines(img, lines, rate=1.0 / 8):
    result = []

    for src_line in lines:
        line = src_line[0]
        rho = line[0]
        if rho < img.shape[1] * rate:
            continue

        result.append(src_line)

    return result


def equalizeHist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output


def equalizeAdaptiveHist(img, grid_side=10):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_side, grid_side))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img


def drawLines(img, lines, color=(0, 200, 0), thickness=3):
    canvas = img.copy()
    for line in lines:

        line = line[0]

        rho = line[0]
        theta = line[1]
        a = m.cos(theta)
        b = m.sin(theta)
        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        # print((rho, theta), (x0, y0), pt1, pt2)

        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    return canvas
