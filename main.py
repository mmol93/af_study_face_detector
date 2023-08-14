import numpy as np
import cv2, dlib, sys

# init video file
cap = cv2.VideoCapture('video.mp4')
scale = 0.3

# init face detector module
detector = dlib.get_frontal_face_detector()
# set machine learned model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, img = cap.read()

    # break when the frame doesn't exist.
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    original_img = img.copy()

    # detect face
    faces = detector(img)
    face = faces[0]

    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top(),), pt2=(face.right(), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # show movie
    cv2.imshow('img', img)
    cv2.waitKey(1)
