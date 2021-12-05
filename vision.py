import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import cv2
import NN_recog


kcft = cv2.TrackerKCF_create()

train_all, train_all_labels = NN_recog.dataset_proc()
model = NN_recog.model_dev(train_all, train_all_labels)
train_all = []
train_all_labels = []

def show_image(img):
  fig = plt.figure()
  fig.set_size_inches(18, 10) # You can adjust the size of the displayed figure
  plt.imshow(img)



def main():

    #set parameters
    episodes = 1

    extract_statespace(episodes)


    return

#Extracts state information from the images from ML training
def extract_statespace(episodes):

    #Names of the characters fought in each episode
    characters = ['blanka','chunli','dahlism','ehonda','guile','ken','ryu','zangief']
    #characters = ['blanka','chunli','ehonda','ryu','zangief']
    #Data we're extracting
    states = ['x_position', 'y_position', 'status', 'health', 'round_timer']

    #Set up dataframe to store information in
    state_df = pd.DataFrame(columns=states)

    for e in range(0,episodes):
        for c in characters:
            #Load images from numpy matrix
            title = './data/' + 'episode' + str(e) + '_' + c + '_images.npy'
            images = np.load(title)
            #Loop through images
            counter = 0
            init_kcft, frame = True, images[0]
            frame = (np.uint8(frame))
            frame = cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB)
            height, width, layers = images[0].shape
            size = (width, height)

            out = cv2.VideoWriter('project_annotated.avi',cv2.VideoWriter_fourcc(*'DIVX'), 40, size)
            roi = cv2.selectROI(frame, False)
            init_kcft = kcft.init(np.uint8(frame), roi)
            for i in range(1,np.shape(images)[0]):
                img1 = np.uint8(images[i])
                # next frame
                frame = frame = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            
                init_kcft, roi = kcft.update(frame)
                if init_kcft:
                    #  success
                    corner_1 = (int(roi[0]), int(roi[1]))
                    corner_3 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                    cv2.rectangle(frame, corner_1, corner_3, (0,0,0), 3, 2)
                    k = cv2.waitKey(1) & 0xff
                    if k == 27 : break
                else :
                    # In case of failure
                    cv2.putText(frame, "Target Undetectable", (50,100), 
                                cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),1)
                cv2.putText(frame, "KCF Tracker", (50,50), cv2.FONT_HERSHEY_PLAIN, 3, 
                            (255,255,255),1);
                cv2.putText(frame, "ROI", corner_1, cv2.FONT_HERSHEY_PLAIN, 2, 
                            (0,0,255),1);
                #cv2.imshow(frame)
                out.write(cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB))
                cv2.imshow("annotated", cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB))
                #cv2.destroyAllWindows()
            out.release()
            cv2.destroyAllWindows()

            #Test to look at images
            if(i == 0):
                print("Character = " + c + "   Episode = " + str(e))
                fig = plt.figure()
                plt.imshow(img)
                plt.waitforbuttonpress()
                plt.close()

            #Get ROI, status from template match?

            #Get position from KCFT

            #Get timer using OCR
            #time = extractText(img)

            #Get health
            #health = getHealth(img)

            #Add information to dataframe (placeholder values)
            info = [[2.1, 0.5, 'crouch', 0, 99]]   #pos, state, health, timer
            info_df = pd.DataFrame(info, index = [counter], columns=states)
            state_df = state_df.append(info_df)
            counter += 1

        #Save df as csv
        #print("saving")
        #state_df.to_csv('./data/' + 'episode' + str(e) + '_' + c + '_vision_states.csv')

    return

#Estimates player health from the image of the health bar
def getHealth(img):

    #Set region of image where healthbar is
    x1 = 12
    x2 = 24
    y1 = 32
    y2 = 120

    #Get ROI for timer
    roi = img[x1:x2, y1:y2]

    #Search for yellow part of bar
    for c in range(0,y2-y1):
        g = roi[5,c,1]
        #Found yellow edge
        if(g == 204):
            health = 100 - (np.float32(c) / np.float32(y2-y1))*100.0
            return health
    return 0

    return health

#Runs through images in the dataset and reports the timer at each frame
def extractText(img):

    #Set region of image where timer is
    #Exact box around numbers
    #x1 = 29
    #x2 = 40
    #y1 = 120
    #y2 = 135
    #test - wider bounding box
    x1 = 26
    x2 = 43
    y1 = 117
    y2 = 138

    #Test - blur the entire image
    #img = cv2.GaussianBlur(np.float32(img),(3,3),cv2.BORDER_DEFAULT)
    #img = img/np.amax(img)
    #fig = plt.figure()
    #plt.imshow(img)
    #plt.waitforbuttonpress()
    #plt.cla()

    #Get ROI for timer
    roi = img[x1:x2, y1:y2]
    #roi = cv2.cvtColor(np.float32(roi), cv2.COLOR_BGRA2GRAY)

    #Blur the ROI w/ timer
    #roi = cv2.GaussianBlur(np.float32(roi),(3,3),cv2.BORDER_DEFAULT)
    #roi = roi/np.amax(roi)
    #fig = plt.figure()
    #plt.imshow(roi)
    #plt.waitforbuttonpress()
    #plt.close()


    #Test detection w/o thresholding
    #pil_img = Image.fromarray(roi)  #Change from openCV BGR to Pillow RGB
    #if(pil_img.mode != 'RGB'):
#        pil_img = pil_img.convert('RGB')
    #text = pytesseract.image_to_string(pil_img, lang='eng', config='-c tessedit_char_whitelist=0123456789')
    #text = pytesseract.image_to_string(pil_img, lang='eng', config='digits')
    #print("ROI text = " + text)


    #Threshold image for text extraction
    #junk, roi_thresh = cv2.threshold(roi, 65, 255, cv2.THRESH_BINARY)

    #Manually create thresholded image
    red = np.array([232,0,0])               #Colors in the timer numbers
    yellow = np.array([232,204,0])
    orange = np.array([232,100,0])
    roi_thresh = np.zeros(np.shape(roi),np.int32())
    for r in range(0,np.shape(roi)[0]):
        for c in range(0,np.shape(roi)[1]):
            if(np.array_equal(roi[r][c],red) or np.array_equal(roi[r][c],yellow) or np.array_equal(roi[r][c],orange)):
                roi_thresh[r][c] = [0,0,0]
            else:
                roi_thresh[r][c] = [255,255,255]

    #Format image for detection
    roi_thresh = cv2.cvtColor(np.float32(roi_thresh), cv2.COLOR_BGRA2GRAY)
    roi_thresh = roi_thresh/np.amax(roi_thresh)
    roi_thresh = cv2.resize(roi_thresh, dsize=(roi_thresh.shape[1]*4,roi_thresh.shape[0]*4), interpolation=cv2.INTER_AREA)
    roi_thresh = cv2.GaussianBlur(np.float32(roi_thresh),(5,5),cv2.BORDER_DEFAULT)
    junk, roi_thresh = cv2.threshold(roi_thresh, 0.2, 1, cv2.THRESH_BINARY)

    #Extract text
    pil_img = Image.fromarray(roi_thresh)  #Change from openCV BGR to Pillow RGB
    if(pil_img.mode != 'RGB'):
        pil_img = pil_img.convert('RGB')
    text = pytesseract.image_to_string(pil_img, lang='eng', config='-c tessedit_char_whitelist=0123456789 --psm 6')
    #text = pytesseract.image_to_string(pil_img, lang='eng', config='digits')
    print(text)

    #Test - view image
    #print("Thresholded text = " + text)
    fig = plt.figure()
    plt.title(text)
    plt.imshow(roi_thresh, cmap='gray')
    plt.waitforbuttonpress()
    plt.close()

    return 0



#Here we'll compare the accuracy of our predictions to the values from memory
def experiment():

    #Names of the characters fought in each episode
    characters = ['blanka','chunli','dahlism','ehonda','guile','ken','ryu','zangief']

    for e in range(0,episodes):
        for c in characters:
            #Load memory csv into pandas dataframe
            memtitle = './data/' + 'episode' + str(e) + '_' + c + '_statespace.csv'
            memory_df = pd.read_csv(memtitle)
            #Load vision csv into pandas dataframe
            vistitle = './data/' + 'episode' + str(e) + '_' + c + '_vision_states.csv'
            vision_df = pd.read_csv(vistitle)

            #Find error between our estimates and the memory values
                #Most values we can get the L2 norm or something
                #Need to present character status in different graph (recall/precision?)

            #save results to graph in array

    #Use matplotlib to present information across n episodes in one figure

    return



if __name__ == "__main__":
    main()
