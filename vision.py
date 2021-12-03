import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import cv2

def main():

    #set parameters
    episodes = 1

    extract_statespace(episodes)


    return

#Extracts state information from the images from ML training
def extract_statespace(episodes):

    #Names of the characters fought in each episode
    characters = ['blanka','chunli','dahlism','ehonda','guile','ken','ryu','zangief']
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
            for i in range(0,np.shape(images)[0]):
                img = images[i]
                #Test to show images are being loaded correctly
                if(i == 0):
                    print("Character = " + c + "   Episode = " + str(e))
                    fig = plt.figure()
                    plt.imshow(img)
                    plt.waitforbuttonpress()
                    plt.cla()

                    #Get initial location of Ryu

                #Get position from KCFT

                #Get character status

                #Get timer using OCR
                #time = extractText(img)

                #Get health
                health = getHealth(img)

                #Add information to dataframe (placeholder values)
                info = [[2.1, 0.5, 'crouch', health, 180]]   #pos, state, health, timer
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

    #Get ROI for timer (grayscale)
    roi = img[x1:x2, y1:y2]
    #roi_color = roi
    #roi = cv2.cvtColor(np.float32(roi), cv2.COLOR_BGRA2GRAY)
    #roi = roi/np.amax(roi)

    #print("Red part = ")
    #print(roi_color[5,20])
    #print("Yellow part = ")
    #print(roi_color[5,70])

    #Red = [232 0 0]
    #Yellow = [232 204 0]

    #showim = 0
    #if(showim == 1):
    #    fig = plt.figure()
    #    ax1 = fig.add_subplot(2,2,1)
    #    plt.imshow(roi,cmap='gray')
    #    ax2 = fig.add_subplot(2,2,2)
    #    ax2.imshow(roi_color)
    #    plt.waitforbuttonpress()
    #    plt.close()

    #Search for yellow part of bar
    for c in range(0,y2-y1):
        g = roi[5,c,1]
        #Found yellow edge
        if(g == 204):
            health = 100 - (np.float32(c) / np.float32(y2-y1))*100.0
            return health
    return 0

    print(health)
    titlegraph = "Health = " + str(health)
    fig = plt.figure()
    plt.title(titlegraph)
    plt.imshow(roi)
    plt.waitforbuttonpress()
    plt.close()


    return health

#Runs through images in the dataset and reports the timer at each frame
def extractText(img):

    #Set region of image where timer is
    #Exact box around numbers
    x1 = 29
    x2 = 40
    y1 = 120
    y2 = 135
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

    #Get ROI for timer (grayscale)
    roi = img[x1:x2, y1:y2]
    roi = cv2.cvtColor(np.float32(roi), cv2.COLOR_BGRA2GRAY)

    #Blur the ROI w/ timer
    roi = cv2.GaussianBlur(np.float32(roi),(3,3),cv2.BORDER_DEFAULT)
    roi = roi/np.amax(roi)
    fig = plt.figure()
    plt.imshow(roi,cmap='gray')
    plt.waitforbuttonpress()
    plt.close()


    #Test detection w/o thresholding
    pil_img = Image.fromarray(roi)  #Change from openCV BGR to Pillow RGB
    if(pil_img.mode != 'RGB'):
        pil_img = pil_img.convert('RGB')
    #text = pytesseract.image_to_string(pil_img, lang='eng', config='-c tessedit_char_whitelist=0123456789')
    text = pytesseract.image_to_string(pil_img, lang='eng', config='digits')
    print("ROI text = " + text)


    #Threshold image for text extraction
    junk, roi_thresh = cv2.threshold(roi, 65, 255, cv2.THRESH_BINARY)

    #Extract text
    roi_thresh = cv2.bitwise_not(roi_thresh)    #Invert color so text is black
    pil_img = Image.fromarray(roi_thresh)  #Change from openCV BGR to Pillow RGB
    if(pil_img.mode != 'RGB'):
        pil_img = pil_img.convert('RGB')
    text = pytesseract.image_to_string(pil_img, lang='eng', config='-c tessedit_char_whitelist=0123456789')

    #Test - view image
    print("Thresholded text = " + text)
    fig = plt.figure()
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
