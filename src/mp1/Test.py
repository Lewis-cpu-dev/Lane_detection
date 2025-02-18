from PIL import Image
import cv2

# img = Image.open("/opt/data/TUSimple/train_set/clips/0313-1/60/20.jpg")
# img.show()
imgc =cv2.imread("/opt/data/TUSimple/train_set/clips/0313-1/60/20.jpg") 
cv2.imshow("test",imgc)
cv2.waitKey(0)

