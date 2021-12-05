import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import cv2
import NN_recog
import templateMatching as tm




train_all, train_all_labels = NN_recog.dataset_proc()
model = NN_recog.model_dev(train_all, train_all_labels)
train_all = []
train_all_labels = []
#train_labels = ['block', 'crouch', 'idle', 'walking', 'jump', 'punch', 'kick']
train_labels = ['kick', 'punch', 'jump', 'walking', 'idle', 'block', 'crouch']
def show_image(img):
  fig = plt.figure()
  fig.set_size_inches(18, 10) # You can adjust the size of the displayed figure
  plt.imshow(img)



def main():

    #set parameters
    episodes = 1
    extract_statespace(episodes)
    #experiment(episodes)

    return

#Extracts state information from the images from ML training
def extract_statespace(episodes):

    #Names of the characters fought in each episode
    #characters = ['blanka','chunli','dahlism','ehonda','guile','ken','ryu','zangief']
    characters = ['blanka']
    #Data we're extracting
    states = ['kcft_x_position', 'kcft_y_position', 'template_x_position', 'template_y_position', 'nn_status', 'template_status', 'validation_status','health', 'round_timer']

    templates = tm.loadTemplates('./assets/templates')

    for e in range(0,episodes):
        for c in characters:
            print('Now Extracting ', c, ' Episode ', e)
            #Load images from numpy matrix
            title = './assets/data/' + 'episode' + str(e) + '_' + c
            images = np.load(title + '_images.npy')
            #Set up dataframe to store information in
            state_df = pd.DataFrame(columns=states)
            #Load validation dataset
            val_df = pd.read_csv(title + '_statespace.csv',usecols=['Action'])
            #Loop through images
            counter = 0
            init_kcft, frame = True, np.uint8(images[0])
            #size = (frame.shape[0], frame.shape[1])
            size = (250,200)
            #out = cv2.VideoWriter('project_mod.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
            out = cv2.VideoWriter('project_annotated_'+c+'_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 40, size)
            # Gets initial image and initializes KCFT
            tm_label, roi = tm.getMatchLocation(np.uint8(frame),templates)
            kcft = cv2.TrackerKCF_create()
            init_kcft = kcft.init(frame, roi)

            #Iterate through images
            for i in range(1,np.shape(images)[0]):
                # next frame
                frame = np.uint8(images[i])
                #Get character action & location from template match
                match_label,match_loc = tm.getMatchLocation(frame,templates)
                # returns topleftX,topleftY,width, height
                init_kcft, roi = kcft.update(frame)
                if init_kcft:
                    #  success
                    corner_1 = (int(roi[0]), int(roi[1]))
                    corner_3 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                    cv2.rectangle(frame, corner_1, corner_3, (0,0,0), 3, 2)
                    # Checks if current label is able to be predicted by NN
                    for label in train_labels:
                        if label in val_df['Action'][i]:
                            gr_frame = cv2.cvtColor(frame[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]], cv2.COLOR_RGB2GRAY)
                            nn_status = NN_recog.pred_loop(gr_frame, model)
                            put = "ROI & Action:" + str(nn_status)
                            break;
                        else:
                            put = "Action Undefined"
                            nn_status = "N/A"
                else:
                    # check with template matching if KCFT cant detect
                    match_label, roi = tm.getMatchLocation(frame, templates)
                    kcft = cv2.TrackerKCF_create()
                    init_kcft = kcft.init(np.uint8(frame), roi)
                    init_kcft, roi = kcft.update(frame)
                    if init_kcft:
                        #  success
                        corner_1 = (int(roi[0]), int(roi[1]))
                        corner_3 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                        cv2.rectangle(frame, corner_1, corner_3, (0,0,0), 3, 2)
                        # Checks if current label is able to be predicted by NN
                        for label in train_labels:
                            if label in val_df['Action'][i]:
                                gr_frame = cv2.cvtColor(frame[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]], cv2.COLOR_RGB2GRAY)
                                nn_status = NN_recog.pred_loop(gr_frame, model)
                                put = "ROI & Action:" + str(nn_status)
                                break;
                            else:
                                put = "Action Undefined"
                                nn_status = "N/A"
                    else:
                        # In case of failure
                        cv2.putText(frame, "Target Undetectable", (50,100),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),1)
                        put = "ROI Undefined"
                        nn_status = "undefined"
                cv2.putText(frame, "KCF Tracker", (50,50), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255,255,255),1);
                cv2.putText(frame, put, corner_1, cv2.FONT_HERSHEY_PLAIN, 2,
                            (0,0,255),1);
                #cv2.imshow("annotated", frame)

                #Get position and status from template match
                template_x_pos = np.int32((2*match_loc[0]+match_loc[2])/2.0)
                template_y_pos = np.int32((2*match_loc[1]+match_loc[3])/2.0)
                template_status = match_label

                #Get position from KCFT
                kcft_x_pos = np.int32((roi[0]+roi[2])/2.0)
                kcft_y_pos = np.int32((roi[1]+roi[3])/2.0)

                #Get timer using template match
                time = getTimer(frame)

                #Get health
                health = getHealth(frame)

                #Add information to dataframe
                info = [[kcft_x_pos, kcft_y_pos, template_x_pos, template_y_pos, nn_status, template_status, val_df['Action'][i], health, time]]   #pos, state, health, timer
                info_df = pd.DataFrame(info, index = [counter], columns=states)
                state_df = state_df.append(info_df)
                counter += 1

                #cv2.destroyAllWindows()
                out.write(cv2.cvtColor(cv2.resize(frame, size,  interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            out.release()
            #cv2.destroyAllWindows()

            #Save df as csv
            print("Saving ", c, ' Episode ', e)
            state_df.to_csv('./results/' + 'episode' + str(e) + '_' + c + '_vision_states.csv')

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

#Saves thresholded numbers (0-9) as templates
def saveNumberTemplate():

    #Load images from numpy matrix
    title = './data/episode0_blanka_images.npy'
    images = np.load(title)

    x1 = 29
    x2 = 40
    y1 = 120
    y2 = 135

    numb = 9
    for i in range(0,np.shape(images)[0]):
        img = images[i]
        roi = img[x1:x2, y1:y2]
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
        #show/save images as templates
        if(i % 40 == 0 and i < 370):
            roi_thresh = roi_thresh[:,8:]
            cv2.imwrite('./assets/numbers/'+str(numb)+'_thresh.png',roi_thresh)
            numb -= 1

    for i in range(0,10):
        title = './assets/numbers/' + str(i)+'_thresh.png'
        img = np.asarray(Image.open(title))
        #img = img.astype("float32")/255.0
        fig = plt.figure()
        plt.imshow(img)
        plt.waitforbuttonpress()
        plt.close()

    return

#Uses template matching to read the time remaining for the match
def getTimer(img):

    x1 = 29
    x2 = 40
    y1 = 120
    y2 = 135

    #Get roi for both digits
    roi = img[x1:x2, y1:y2]

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

    left_roi = roi_thresh[:, :7]
    right_roi = roi_thresh[:, 8:]

    #Try to match both numbers
    left_num = 0
    right_num = 0
    for i in range(0,10):
        title = './assets/numbers/' + str(i) + '_thresh.png'
        img = np.asarray(Image.open(title))
        #img = img.astype("float32")/255.0
        if(np.array_equal(img,left_roi)):
            left_num = i
        if(np.array_equal(img,right_roi)):
            right_num = i

    time = 10*left_num + right_num

    return time

#Runs through images in the dataset and attempts to use OCR recognition. Doesn't work.
def ocrText(img):

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

    #Get ROI for timer
    roi = img[x1:x2, y1:y2]

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
def experiment(episodes):

    #Names of the characters fought in each episode
    #characters = ['blanka','chunli','dahlism','ehonda','guile','ken','ryu','zangief']
    characters = ['blanka']

    for e in range(0,episodes):
        for c in characters:
            #Load memory csv into pandas dataframe
            memtitle = './data/' + 'episode' + str(e) + '_' + c + '_statespace.csv'
            memory_df = pd.read_csv(memtitle)
            #Load vision csv into pandas dataframe
            vistitle = './data/' + 'episode' + str(e) + '_' + c + '_vision_states.csv'
            vision_df = pd.read_csv(vistitle)
            print("Memory size = %d" % memory_df.size)
            print("Vision size = %d" % vision_df.size)

            #Find error between our estimates and the memory values
                #Most values we can get the L2 norm or something
                #Need to present character status in different graph (recall/precision?)

            #save results to graph in array

    #Use matplotlib to present information across n episodes in one figure

    return



if __name__ == "__main__":
    main()
