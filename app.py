import mediapipe as mp
import cv2
import random
import time


def main() -> None:
    print("[INFO] main() is called")
    # ref: https://google.github.io/mediapipe/solutions/face_detection.html#overview

    mp_face_detection = mp.solutions.face_detection
    mp_drawing_util = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)  # 接続されているデバイスの0番目．認識されているとは限らない
    print("[INFO] camera device connected")

    with mp_face_detection.FaceDetection(  # withで開くことで，close処理を忘れない
        model_selection=1,  # 0:short-range(within 2[m]) / 1:full-range(within 5[m])
        min_detection_confidence=0.5  # ゼミでやったconfidence値
    ) as fd:
        while cap.isOpened():
            status, image = cap.read()
            if not status:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # eng:Flip the image horizontally for a selfie-view display.
            # ja:cv2.flip(image, 1)で左右反転して鏡の様にしている
            image = cv2.flip(image, 1)
            image_height, image_width, _ = image.shape  # 後で使うので取得しておく

            #
            cv2.putText(
                img=image,
                org=(5, 30),  # (lx,ly);左下頂点のx,y座標
                text="this is a fake thermo metor :)",
                color=(0, 255, 0),  # (r,g,b)
                fontScale=0.5,
                thickness=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                lineType=cv2.LINE_AA,
            )

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ConVerT color (bgr -> rgb)
            results = fd.process(image)  # main process
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # re ConVerT color (rgb -> bgr)

            # Draw the face detection annotations on the image.
            if results.detections:
                # 顔が1つ以上検出された場合
                for detection in results.detections:
                    # mark up face
                    mp_drawing_util.draw_detection(image, detection)

                    # draw fake temperature
                    lx: float = detection.location_data.relative_bounding_box.xmin * image_width
                    ly: float = detection.location_data.relative_bounding_box.ymin * image_height + \
                        detection.location_data.relative_bounding_box.height * image_height
                    cv2.putText(
                        img=image,
                        org=(int(lx), int(ly)),  # (lx,ly);左下頂点のx,y座標
                        text="{:2.1f}[degree celsius]".format(random.uniform(36, 38)),  # uniform(a,b) -> [a,b]
                        color=(0, 255, 0),  # (r,g,b)
                        fontScale=0.8,
                        thickness=1,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        lineType=cv2.LINE_AA,
                    )

            # ウィンドウサイズを可変にする．第一引数のidはimshowと一致させる必要がある(飾りでは無い)
            cv2.namedWindow("Fake Thermometor", cv2.WINDOW_NORMAL)
            cv2.imshow("Fake Thermometor", image)

            key = cv2.waitKey(5)  # 5[msec]の間入力待機
            if key == 27:  # key == [ESC]
                print("[INFO] detect [ESC] key pressed. stop process..")
                break

    # close処理
    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("[INFO] total process time:{:.3f}[sec]".format((end - start)))
