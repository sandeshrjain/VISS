# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:48:04 2021

@author: Sandesh Jain
"""



import cv2    

def tracker(vid_name):
    kcft = cv2.TrackerKCF_create()
    curve=[]
    vid = cv2.VideoCapture(vid_name)
    init_kcft, frame = vid.read()    
    roi = cv2.selectROI(frame, False)
    init_kcft = kcft.init(frame, roi)
    while True:
            # next frame
            init_kcft, frame = vid.read()
            if not init_kcft:
                break
            # Update 
            init_kcft, roi = kcft.update(frame)
            if init_kcft:
                #  success
                corner_1 = (int(roi[0]), int(roi[1]))
                corner_3 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                cv2.rectangle(frame, corner_1, corner_3, (0,0,0), 3, 2)
                centroid = ((int(roi[0] + (roi[2])/2), int(roi[1] + (roi[3])/2)))
                curve.append(centroid)
            else :
                # In case of failure
                cv2.putText(frame, "Target Undetectable", (50,100), 
                            cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),1)
            cv2.putText(frame, "KCF Tracker", (50,50), cv2.FONT_HERSHEY_PLAIN, 3, 
                        (255,255,255),1);
            cv2.putText(frame, "ROI", corner_1, cv2.FONT_HERSHEY_PLAIN, 2, 
                        (0,0,255),1);
            
            for point in range(1, len(curve)):
                cv2.line(frame, curve[point-1], curve[point], (255,0,0), 1) 
            # Display result
            cv2.imshow("Tracking", frame)
    
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
    path = './tracker_plot/kcft_curve.jpg'
    cv2.imwrite(path , frame)
    cv2.destroyAllWindows()

#test block

tracker("street_fighter.webm")
