import cv2

cap = cv2.VideoCapture(2)

while True:
    _, frame = cap.read()
    
    cv2.imshow("Testing", frame)
    if cv2.waitKey(1) == ord('q'):
        break

