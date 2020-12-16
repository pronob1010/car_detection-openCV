import cv2

# capture frames from a video
cap = cv2.VideoCapture(r'F:\opencv\Car-Detection-Basic-Open-CV\carv2.mp4')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier(r'F:\opencv\Car-Detection-Basic-Open-CV\carx.xml')

# loop runs if capturing has been initialized.
while True:
    ret, frames = cap.read()
    frames_color_convert_gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(frames_color_convert_gray, 1.1, 2)
    print(cars)
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('test-1', frames )
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
