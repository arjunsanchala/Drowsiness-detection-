
# Import packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

def Calculate_eye_aspect_ratio(eye):
    # compute the distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])

    # The eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

# EAR must be below the threshold
EYE_treshold = 0.35
EYE_FRAMES = 3

# Variables
COUNTER = 0
TOTAL = 0
listEAR = []

# face detector based of HOG and then create the facial landmark predictor
print("----- Loading facial landmark predictor -----")
detector = dlib.get_frontal_face_detector()

# Put your file path
predictor = dlib.shape_predictor("/Users/aj/Downloads/shape_predictor_68_face_landmarks.dat")

(lEStart, lEEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rEStart, rEEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

stream = VideoStream(src=0).start()
time.sleep(1.0)

elapsed_time = []
start = time.time()
elapse = 0

# loop over video frames
while True:

    frames = stream.read()
    frames = imutils.resize(frames, width=1000)
    gray_image = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # detect faces
    rects = detector(gray_image, 0)

    for rect in rects:
        # determine the facial landmarks for the face region
        shape_p = predictor(gray_image, rect)
        shape_p = face_utils.shape_to_np(shape_p)

        # compute the eye aspect ratio for both eyes
        leftEye = shape_p[lEStart:lEEnd]
        rightEye = shape_p[rEStart:rEEnd]

        left_EAR = Calculate_eye_aspect_ratio(leftEye)
        right_EAR = Calculate_eye_aspect_ratio(rightEye)

        ear = (left_EAR + right_EAR)

        # visualisation of eyes
        leftEye_Hull = cv2.convexHull(leftEye)
        rightEye_Hull = cv2.convexHull(rightEye)
        cv2.drawContours(frames, [leftEye_Hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frames, [rightEye_Hull], -1, (0, 255, 0), 1)


        # check to see if the eye aspect ratio is below threshold,
        # and if so, detect the duration of drowsiness


        if ear < EYE_treshold:

            elapsed_time.append(time.time())
            if len(elapsed_time)<2:
                elapse =  elapse + (elapsed_time[-1] - elapsed_time[0])

            else:
                elapse = elapse + (elapsed_time[-1] - elapsed_time[-2])
            cv2.putText(frames, "You have closed your eyes for {:.2f} seconds".format(elapse), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frames, "Eye ratio: {:.2f}".format(ear), (770, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            listEAR.append(round(ear,2))


        else:

            elapsed_time.clear()
            cv2.putText(frames, "You have closed your eyes for {:.2f} seconds".format(elapse), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0 , 0), 2)
            cv2.putText(frames, "Eye ratio: {:.2f}".format(ear), (770, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            listEAR.append(round(ear,2))

    # The frame
    cv2.imshow("Frame", frames)
    key = cv2.waitKey(1) & 0xFF

    # To break the loop, press 'q'
    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
stream.stop()
print("You have closed your eyes for {:.2f} seconds.".format(elapse))
# print(listEAR)
