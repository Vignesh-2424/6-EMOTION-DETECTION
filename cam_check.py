import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open camera")
else:
    print("✅ Camera is working")
cap.release()
