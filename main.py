import cv2, numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# define constants for eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
def get_EAR(eye_landmarks):
    # calculate the euclidean distances between the two sets of 
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # calculate the euclidean distance between the horizontal
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    return (A + B) / (2 * C)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        p1 = np.array((landmarks.part(36).x, landmarks.part(36).y))
        p2 = np.array((landmarks.part(37).x, landmarks.part(37).y))
        p3 = np.array((landmarks.part(38).x, landmarks.part(38).y))
        p4 = np.array((landmarks.part(39).x, landmarks.part(39).y))
        p5 = np.array((landmarks.part(40).x, landmarks.part(40).y))
        p6 = np.array((landmarks.part(41).x, landmarks.part(41).y))

        eye_landmarks = [p1, p2, p3, p4, p5, p6]
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)

        # compute the eye aspect ratio for the right eye
        right_eye_aspect_ratio = get_EAR(eye_landmarks)

        # if the eye aspect ratio is below the blink threshold,
        # increment the blink frame counter
        if right_eye_aspect_ratio < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and increment the total
        # number of blinks
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(right_eye_aspect_ratio), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        hor_line = cv2.line(frame, eye_landmarks[0], eye_landmarks[3], (0,255,0), 2)
        ver_line1 = cv2.line(frame, eye_landmarks[1], eye_landmarks[5], (0,255,0), 2)
        ver_line2 = cv2.line(frame, eye_landmarks[2], eye_landmarks[4], (0,255,0), 2)
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows() 