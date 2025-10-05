from apriltag_pose_estimation.core import AprilTagDetector
import cv2


def main() -> None:

    detector = AprilTagDetector()
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow('camera')
    try:
        while True:
            not_closed, frame = video_capture.read()
            if not not_closed:
                return
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(img_gray)
            for detection in detections:
                if detection.tag_id == 0:
                    # image width is an int
                    image_width = img_gray.shape[1] / 2
                    horizontal_distance = detection.center[0] - image_width
                    if horizontal_distance < 0:
                        print(f"The image is left of the center. The value is {horizontal_distance}")
                    elif horizontal_distance > 0:
                        print(f"The image is right of the center. The value is {horizontal_distance}")
                    
                    
            cv2.imshow('camera', frame)
            cv2.waitKey(1)
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

