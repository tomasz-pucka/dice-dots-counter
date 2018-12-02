import numpy as np
import cv2
import urllib.request as ur

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def countDots(dice):
    size = 200
    dice = cv2.resize(dice, (size,size))

    cv2.floodFill(dice, None, (0,0), 255)
    cv2.floodFill(dice, None, (0,size-1), 255)
    cv2.floodFill(dice, None, (size-1,0), 255)
    cv2.floodFill(dice, None, (size-1,size-1), 255)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = True
    params.minInertiaRatio = 0.35
    params.filterByConvexity = True
    params.minConvexity = 0.3
    params.filterByArea = True
    params.minArea = (size*size)*0.008
    #params.filterByCircularity = True
    #params.minCircularity = 0.7

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dice)
    return len(keypoints)

url = 'http://192.168.0.13:8080/shot.jpg'
window_names = ['Normal', 'Result']
cv2.namedWindow(window_names[0], cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_names[0], 683, 385)
cv2.namedWindow(window_names[1], cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_names[1], 683, 385)
#cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):

    #capture from smartphone camera
    #''' uncomment to set capture from smartphone camera
    resp = ur.urlopen(url)
    img_np = np.asarray(bytearray(resp.read()), dtype="uint8")
    img_original = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    #'''

    #capture from webcam
    #ret, frame = cap.read() #uncomment to set capture from camera
    #img_original = frame   #uncomment to set capture from camera

    img_gamma = gamma_correction(img_original, 0.2)
    img_gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2,2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    cv2.imshow(window_names[0], img)
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        diceArea = cv2.contourArea(cnt)
        if (diceArea > 1500 and diceArea < 50000):
            x, y, w, h = cv2.boundingRect(cnt)
            if(h > w+40 or h < w-40): continue
            diceROI = img[y:y+h, x:x+w]
            dotsCount = countDots(diceROI)
            if (dotsCount > 0 and dotsCount <= 6):
                img_original = cv2.rectangle(img_original, (x, y), (x + w, y + h), (100, 100, 255), 2)
                cv2.putText(img_original, str(dotsCount), (x, y+h+40), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow(window_names[1], img_original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
cv2.destroyAllWindows()