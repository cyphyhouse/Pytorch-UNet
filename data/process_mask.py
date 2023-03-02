import cv2 
import os 


if __name__ == "__main__":
    for fn in os.listdir('./data/masks'):
        cv_image = cv2.imread('./data/masks/'+fn)
        tmp = fn.split('.')
        # cv_image_comp = cv2.imread('./data/masks/1ae8a68a40e4_05_mask.gif')
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        mask_red_1 = cv2.inRange(hsv, (0, 50, 20), (5, 255, 255))
        mask_red_2 = cv2.inRange(hsv, (175, 50, 20), (180 ,255, 255))

        mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
        img_red = cv2.bitwise_and(cv_image, cv_image, mask=mask_red)
        img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(img_red_gray, 10, 255,cv2.THRESH_BINARY)[1]
        cv2.imshow('aaa',img_binary)
        cv2.imwrite('./data/label/'+tmp[0]+'.png',img_binary)
        cv2.waitKey(1)
        # continue
