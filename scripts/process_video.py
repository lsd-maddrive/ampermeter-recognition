import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(PROJECT_ROOT, "notebooks"))

import lib


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Process video file")
    parser.add_argument("-c", "--config", required=True, help="path to config file")
    parser.add_argument(
        "-i", "--input", required=True, help="path to video file to process"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="path to result video file"
    )

    args = parser.parse_args()
    return args


def read_config(fpath):
    import yaml

    with open(fpath) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config


def main():
    args = get_args()

    config = read_config(args.config)

    input_path = args.input
    output_path = args.output

    reader = lib.VideoReader(input_path)

    # Preparations part
    frame = reader[0]
    roi_poly_pnts = config["objects"]["src_bbox"]

    units_points = config["unit_measures"]
    units_points_x = units_points["x"]
    units_points_y = units_points["y"]

    units_convertor = lib.AngleUnitsConvertor()
    units_convertor.fit(X=units_points_x, y=units_points_y)

    screen_extr = lib.IndicatorScreenExtractor(roi_pnts=roi_poly_pnts, ind_size=400)
    screen_extr.fit(frame)

    arr_det = lib.ArrowDetector()

    text_writer = lib.ImageTextWriter()
    text_writer.fontColor = (0, 170, 0)
    text_writer.fontScale = 2
    text_writer.thickness = 5

    # Processing part
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
        fps=reader.fps,
        frameSize=reader.size,
    )
    reader.move_2_frame(0)

    while True:
        frame = reader.read_frame()
        if frame is None:
            break

        screen_frame = screen_extr.transform(frame)

        _, arrow_angles = arr_det.transform(screen_frame)
        arrow_angles_deg = np.rad2deg(arrow_angles)
        lib.normalize_angles_deg(arrow_angles_deg, copy=False)

        # screen_canvas = lib.drawLines(screen_frame, lines)

        mean_angle = np.mean(arrow_angles_deg)
        mean_unit = units_convertor.transform([mean_angle])[0]

        text_writer.putText(
            frame,
            f"{round(float(mean_angle), 2)} | {round(float(mean_unit), 2)}",
            (10, 70),
        )

        # cv2.imshow("Result", frame)
        # cv2.waitKey(0)

        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

        # break

    video_writer.release()


if __name__ == "__main__":
    main()
