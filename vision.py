import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                #Get timer and health

                #Add information to dataframe (placeholder values)
                info = [[2.1, 0.5, 'crouch', 100, 180]]   #pos, state, health, timer
                info_df = pd.DataFrame(info, index = [counter], columns=states)
                state_df = state_df.append(info_df)
                counter += 1

            #Save df as csv
            print("saving")
            state_df.to_csv('./data/' + 'episode' + str(e) + '_' + c + '_vision_states.csv')

    return

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
