import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

camera = cv2.VideoCapture(0)
timestamp = 0

def count_fingers(landmarks):
    fingers = 0

    if landmarks[4].x > landmarks[3].x:
        fingers += 1

    if landmarks[8].y < landmarks[6].y:
        fingers += 1

    if landmarks[12].y < landmarks[10].y:
        fingers += 1

    if landmarks[16].y < landmarks[14].y:
        fingers += 1

    if landmarks[20].y < landmarks[18].y:
        fingers += 1

    return fingers


with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        operator = None

        if result.hand_landmarks:

            for i, landmarks in enumerate(result.hand_landmarks):

                hand_type = result.handedness[i][0].category_name

                fingers = count_fingers(landmarks)

                # RIGHT HAND = OPERATOR
                if hand_type == "Left":

                    if fingers == 0:
                        operator = "+"

                    elif fingers == 1:
                        operator = "-"

                    elif fingers == 2:
                        operator = "x"

                    elif fingers == 3:
                        operator = "/"

                    elif fingers == 4:
                        operator = "="

                # Draw landmarks
                for landmark in landmarks:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

        if operator is not None:

            cv2.putText(
                frame,
                "Operator: " + operator,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        else:

            cv2.putText(
                frame,
                "Show RIGHT hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "AI Math Solver - Operator Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()