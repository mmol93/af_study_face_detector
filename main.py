import numpy as np
import cv2, dlib, sys

# init video file
cap = cv2.VideoCapture('video.mp4')
scale = 0.3

# load overlay image
overlay_img = cv2.imread('android_logo_512px.png', cv2.IMREAD_UNCHANGED)

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

    # detect face
    faces = detector(img)
    if len(faces) > 0:
        face = faces[0]

    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # compute center of face
    # check if any face is detected
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)
    center_x, center_y = np.mean(shape_2d, axis=0).astype(int)

    # compute face size (resize for image)
    face_size = int(max(bottom_right - top_left) * 1.5)

    # visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top(),), pt2=(face.right(), face.bottom()), color=(255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # show video
    cv2.imshow('img', img)

    cv2.waitKey(1)
