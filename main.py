import cv2
import dlib
import numpy as np

# init video file
cap = cv2.VideoCapture('video.mp4')
scale = 0.3
face_size = 0
pre_face_size = 0

# load overlay image
overlay_img = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)


# image overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


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
    if face_size == 0 or face_size > pre_face_size:
        pre_face_size = face_size
        face_size = int(max(bottom_right - top_left) * 1.5)

    # overlay image
    result = overlay_transparent(img, overlay_img, center_x, center_y - 25, overlay_size=(face_size, face_size))

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
    cv2.imshow('result', result)
    cv2.waitKey(1)
