import cv2
import mediapipe as mp
import time
import re
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("AI Math Solver Started!")
print("Press Q to quit.")


# ============================================================
# VARIABLES
# ============================================================

expression = ""
answer = None

STABLE_TIME = 2.0

stable_left_symbol = None
left_start_time = None
left_accepted = False

stable_operator = None
operator_start_time = None
operator_accepted = False


# ============================================================
# DISTANCE BETWEEN TWO LANDMARKS
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# FINGER STATES
# ============================================================

def get_finger_states(landmarks):

    # --------------------------------------------------------
    # Calculate hand size
    # --------------------------------------------------------

    hand_size = distance(
        landmarks[0],
        landmarks[9]
    )

    if hand_size == 0:
        hand_size = 0.1


    # --------------------------------------------------------
    # THUMB DETECTION
    #
    # Instead of only checking X position, use distance.
    # This works better for 👍 and 🤙.
    # --------------------------------------------------------

    thumb_distance = distance(
        landmarks[4],
        landmarks[5]
    )

    thumb = thumb_distance > hand_size * 0.55


    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index = (
        landmarks[8].y <
        landmarks[6].y
    )


    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    middle = (
        landmarks[12].y <
        landmarks[10].y
    )


    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    ring = (
        landmarks[16].y <
        landmarks[14].y
    )


    # --------------------------------------------------------
    # LITTLE
    # --------------------------------------------------------

    little = (
        landmarks[20].y <
        landmarks[18].y
    )


    return thumb, index, middle, ring, little


# ============================================================
# COUNT NORMAL FINGERS 0-5
# ============================================================

def count_fingers(landmarks):

    fingers = 0

    # Thumb
    if landmarks[4].x > landmarks[3].x:
        fingers += 1

    # Index
    if landmarks[8].y < landmarks[6].y:
        fingers += 1

    # Middle
    if landmarks[12].y < landmarks[10].y:
        fingers += 1

    # Ring
    if landmarks[16].y < landmarks[14].y:
        fingers += 1

    # Little
    if landmarks[20].y < landmarks[18].y:
        fingers += 1


    # Special rule for 4
    # Four fingers up + thumb folded

    if (
        landmarks[8].y < landmarks[6].y
        and landmarks[12].y < landmarks[10].y
        and landmarks[16].y < landmarks[14].y
        and landmarks[20].y < landmarks[18].y
        and landmarks[4].y > landmarks[3].y
    ):
        return 4


    return fingers


# ============================================================
# LEFT HAND
#
# PHYSICAL LEFT HAND:
#
# 👍  = (
# 🤙  = )
#
# Other gestures = numbers
# ============================================================

def get_left_hand_symbol(landmarks):

    thumb, index, middle, ring, little = \
        get_finger_states(landmarks)


    # ========================================================
    # PARENTHESES
    # ========================================================

    # 👍
    # Thumb ONLY = (
    if (
        thumb
        and not index
        and not middle
        and not ring
        and not little
    ):
        return "("


    # 🤙
    # Thumb + Little = )
    if (
        thumb
        and not index
        and not middle
        and not ring
        and little
    ):
        return ")"


    # ========================================================
    # NUMBERS 0-5
    # ========================================================

    fingers = count_fingers(landmarks)

    if 0 <= fingers <= 5:

        return str(fingers)


    return None


# ============================================================
# RIGHT HAND OPERATORS
# ============================================================

def get_operator(fingers):

    if fingers == 0:
        return "+"

    elif fingers == 1:
        return "-"

    elif fingers == 2:
        return "x"

    elif fingers == 3:
        return "/"

    elif fingers == 4:
        return "="

    return None


# ============================================================
# CALCULATE EXPRESSION
# ============================================================

def calculate_expression(exp):

    try:

        # Change x to *
        python_expression = exp.replace("x", "*")


        # Only allow valid mathematical characters
        if not re.fullmatch(
            r"[0-9+\-*/(). ]+",
            python_expression
        ):
            return "Invalid"


        # ----------------------------------------------------
        # Check parentheses
        # ----------------------------------------------------

        balance = 0

        for char in python_expression:

            if char == "(":
                balance += 1

            elif char == ")":
                balance -= 1

                if balance < 0:
                    return "Invalid"


        if balance != 0:
            return "Invalid"


        # ----------------------------------------------------
        # Calculate
        # ----------------------------------------------------

        result = eval(
            python_expression,
            {"__builtins__": None},
            {}
        )


        # Convert 30.0 -> 30
        if isinstance(result, float) and result.is_integer():
            result = int(result)


        return result


    except ZeroDivisionError:

        return "Cannot divide by zero"


    except Exception:

        return "Invalid"


# ============================================================
# MAIN PROGRAM
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera.")
            break


        # Mirror camera
        frame = cv2.flip(frame, 1)


        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        timestamp += 33


        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        current_left_symbol = None
        current_operator = None


        # ====================================================
        # DETECT HANDS
        # ====================================================

        if result.hand_landmarks:

            for i, landmarks in enumerate(
                result.hand_landmarks
            ):

                hand_type = (
                    result.handedness[i][0].category_name
                )


                fingers = count_fingers(
                    landmarks
                )


                # ------------------------------------------------
                # PHYSICAL LEFT HAND
                #
                # MediaPipe "Right"
                # ------------------------------------------------

                if hand_type == "Right":

                    current_left_symbol = \
                        get_left_hand_symbol(
                            landmarks
                        )


                # ------------------------------------------------
                # PHYSICAL RIGHT HAND
                #
                # MediaPipe "Left"
                # ------------------------------------------------

                elif hand_type == "Left":

                    current_operator = \
                        get_operator(
                            fingers
                        )


        # ====================================================
        # LEFT HAND STABILITY
        # ====================================================

        if current_left_symbol is not None:

            if current_left_symbol != stable_left_symbol:

                stable_left_symbol = current_left_symbol

                left_start_time = time.time()

                left_accepted = False


            else:

                if (
                    not left_accepted
                    and left_start_time is not None
                    and time.time() - left_start_time
                    >= STABLE_TIME
                ):

                    # --------------------------------------------
                    # Add number or parenthesis
                    # --------------------------------------------

                    if current_left_symbol.isdigit():

                        expression += current_left_symbol

                        answer = None


                    elif current_left_symbol == "(":

                        # Opening parenthesis can be added
                        # at beginning or after an operator

                        if (
                            expression == ""
                            or expression[-1] in "+-x/("
                            or expression.endswith(" ")
                        ):

                            expression += "("

                            answer = None


                    elif current_left_symbol == ")":

                        # Closing parenthesis only after
                        # a number or another closing parenthesis

                        if (
                            expression
                            and (
                                expression[-1].isdigit()
                                or expression[-1] == ")"
                            )
                        ):

                            expression += ")"

                            answer = None


                    left_accepted = True


        else:

            stable_left_symbol = None

            left_start_time = None

            left_accepted = False


        # ====================================================
        # RIGHT HAND OPERATOR STABILITY
        # ====================================================

        if current_operator is not None:

            if current_operator != stable_operator:

                stable_operator = current_operator

                operator_start_time = time.time()

                operator_accepted = False


            else:

                if (
                    not operator_accepted
                    and operator_start_time is not None
                    and time.time() - operator_start_time
                    >= STABLE_TIME
                ):

                    # ==========================================
                    # EQUAL
                    # ==========================================

                    if current_operator == "=":

                        if (
                            expression
                            and (
                                expression[-1].isdigit()
                                or expression[-1] == ")"
                            )
                        ):

                            answer = calculate_expression(
                                expression
                            )


                    # ==========================================
                    # NORMAL OPERATOR
                    # ==========================================

                    else:

                        if expression:

                            if (
                                expression[-1].isdigit()
                                or expression[-1] == ")"
                            ):

                                expression += (
                                    " "
                                    + current_operator
                                    + " "
                                )

                                answer = None


                    operator_accepted = True


        else:

            stable_operator = None

            operator_start_time = None

            operator_accepted = False


        # ====================================================
        # DISPLAY EXPRESSION
        # ====================================================

        cv2.putText(
            frame,
            "Expression: " + expression,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY CURRENT LEFT SYMBOL
        # ====================================================

        cv2.putText(
            frame,
            "Number/Symbol: "
            + (
                current_left_symbol
                if current_left_symbol is not None
                else "-"
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY OPERATOR
        # ====================================================

        cv2.putText(
            frame,
            "Operator: "
            + (
                current_operator
                if current_operator is not None
                else "-"
            ),
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        if answer is not None:

            cv2.putText(
                frame,
                "Answer: " + str(answer),
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3
            )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "LEFT: Numbers / Parentheses",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Thumb = (     Thumb+Little = )",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "RIGHT: +  -  x  /  =",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "AI Math Solver",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("AI Math Solver stopped.")