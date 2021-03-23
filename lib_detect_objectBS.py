import cv2

#create a MOG2 background subtractor MOG2
bkgM = cv2.createBackgroundSubtractorMOG2(500, 51, 1)

# defined filters
kernelOp = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

kernelCl = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))

def backgroundSubtraction(frame,bkg_list):
    #apply MOG2 BStractor to frame
    fgmask = bkgM.apply(frame)
    # Opening (erode->dilate)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernelOp)
    # Closing (dilate -> erode)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernelCl)
    # convert to binary image
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
    # find all contours in frame
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # only identify (x, y, w, h) of satisfied contours
    if len(contours) != 0:
        #print(contours[0].shape)
        for idx in contours:
            approx = cv2.approxPolyDP(idx, 5, 1)
            s = cv2.contourArea(approx)
            x,y,w,h = cv2.boundingRect(approx)
            if (w*h > 900):
                bkg_list.append((x,y,w,h))
    return bkg_list
def remove_list(name_list):
    #remove elements in list
    if(len(name_list)!= 0):
        for i in range(len(name_list)):
            name_list.pop()
