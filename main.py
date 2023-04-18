import cv2 as cv

# open the default camera
vid = cv.VideoCapture(0)

while True:
    # read the frame from the camera
    ret, frame = vid.read()

    # convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # threshold the grayscale image to obtain a binary image
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # display the binary image
    cv.imshow('Binary Image', binary)

    # exit the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close all windows
vid.release()
cv.destroyAllWindows()
