import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#Here we'll compare the accuracy of our predictions to the values from memory
def main():

    #Number of episodes to test
    episodes = 1

    #Names of the characters fought in each episode
    characters = ['blanka','chunli','dahlism','ehonda','guile','ken','ryu','zangief']
    #characters = ['blanka']

    #Track accuracy across all characters
    avg_kcft_pos_acc = 0.0
    avg_temp_pos_acc = 0.0
    avg_health_acc = 0.0
    avg_time_acc = 0.0
    avg_template_acc = 0.0
    avg_nn_acc = 0.0

    #Making histograms across all characters
    hist_kcft_pos = np.array([])
    hist_temp_pos = np.array([])
    hist_health = np.array([])
    hist_time = np.array([])

    for e in range(0,episodes):
        for c in characters:
            #Load memory csv into pandas dataframe
            memtitle = './assets/data/' + 'episode' + str(e) + '_' + c + '_statespace.csv'
            mem_df = pd.read_csv(memtitle)
            #Load vision csv into pandas dataframe
            vistitle = './results/' + 'episode' + str(e) + '_' + c + '_vision_states.csv'
            vis_df = pd.read_csv(vistitle)

            #Ignore first sample of memory dataset and fix indices
            mem_df.drop(index=mem_df.index[0],axis=[0],inplace=True)
            mem_df.reset_index(drop=True,inplace=True)

            #These should match
            #print("Memory size = %d" % len(mem_df.index))
            #print("Vision size = %d" % len(vis_df.index))
            if(len(mem_df.index) != len(vis_df.index)):
                print("Mismatched datasets, something went wrong.")
                return

            #Find error with position estimates
            kcft_pos = vis_df[['kcft_x_position','kcft_y_position']].to_numpy()
            template_pos = vis_df[['template_x_position','template_y_position']].to_numpy()
            actual_pos = mem_df[['x_position','y_position']].to_numpy()
            kcft_pos_mse = np.linalg.norm(kcft_pos-actual_pos) / np.linalg.norm(actual_pos)
            temp_pos_mse = np.linalg.norm(template_pos-actual_pos) / np.linalg.norm(actual_pos)
            print(c + ": kcft pos estimate accuracy = %f" % kcft_pos_mse)
            avg_kcft_pos_acc += kcft_pos_mse
            print(c+ ": Overall template matching pos estimate accuracy = %f" % temp_pos_mse)
            avg_temp_pos_acc += temp_pos_mse

            #Make histogram of errors between individual measurements to compare kcft and template position
            ind_diff_kcft = np.zeros(len(mem_df.index),dtype=np.float64())
            ind_diff_temp = np.zeros(len(mem_df.index),dtype=np.float64())
            for i in range(0,len(mem_df.index)):
                if(np.linalg.norm(actual_pos[i])==0):
                    ind_diff_kcft[i] = 0
                    ind_diff_temp[i] = 0
                else:
                    ind_diff_kcft[i] = np.linalg.norm(kcft_pos[i]-actual_pos[i]) / np.linalg.norm(actual_pos[i])
                    ind_diff_temp[i] = np.linalg.norm(template_pos[i]-actual_pos[i]) / np.linalg.norm(actual_pos[i])
            hist_kcft_pos = np.append(hist_kcft_pos,ind_diff_kcft)
            hist_temp_pos = np.append(hist_temp_pos,ind_diff_temp)

            #Find error with health measurements
            norm_mem_health = mem_df[['health']].to_numpy() / 176.0
            norm_mem_health = norm_mem_health * 100
            vis_health = vis_df[['health']].to_numpy()
            #Fix values in our data
            for i in range(0,vis_health.size):
                if(vis_health[i] == 0):
                    vis_health[i] = -1
            health_mse = np.linalg.norm(vis_health-norm_mem_health) / np.linalg.norm(norm_mem_health)
            print(c+": Overall health estimate accuracy = %f" % health_mse)
            avg_health_acc += health_mse
            ind_diff_health = np.zeros(len(mem_df.index),dtype=np.float64())
            for i in range(0,len(mem_df.index)):
                if(np.linalg.norm(norm_mem_health[i])==0):
                    ind_diff_health[i] = 0
                else:
                    ind_diff_health[i] = np.linalg.norm(vis_health[i]-norm_mem_health[i]) / np.linalg.norm(norm_mem_health[i])
                    #if(ind_diff_health[i] > 1):
                        #print("Vision health = %f" % vis_health[i])
                        #print("Mem health = %f" % norm_mem_health[i])
                    #print("Relative error = %f" % ind_diff_health[i])
                    #print("")
            hist_health = np.append(hist_health,ind_diff_health)


            #Find error with time measurements
            #Timer starts at ~39205 then counts down, 40 fps
            mem_time = mem_df[['round_timer']].to_numpy()
            vis_time = vis_df[['round_timer']].to_numpy()
            adjusted_mem_time = np.zeros(mem_time.size,dtype=np.int32())
            for i in range(0,mem_time.size):
                adjusted_mem_time[i] = 99-((39205-mem_time[i])%40)
            time_rel_err = np.linalg.norm(vis_time - adjusted_mem_time) / np.linalg.norm(adjusted_mem_time)
            print(c+": Overall time estimate accuracy = %f" % time_rel_err)
            avg_time_acc += time_rel_err
            ind_diff_time = np.zeros(len(mem_df.index),dtype=np.float64())
            for i in range(0,len(mem_df.index)):
                if(np.linalg.norm(adjusted_mem_time)==0):
                    ind_diff_time[i] = 0
                else:
                    ind_diff_time[i] = np.linalg.norm(vis_time[i]-adjusted_mem_time[i]) / np.linalg.norm(adjusted_mem_time[i])
                if(ind_diff_time[i] > 1.0):
                    print("memory time = %f" % mem_time[i])
                    print("read time = %f" % vis_time[i])
                    print("adjusted time = %f\n" % adjusted_mem_time[i])
            hist_time = np.append(hist_time,ind_diff_time)

            #State prediction
            #Should probaly ignore 'block','face_hit','hit','stunned','victory','knockdown','knockout' for template matching
            template_correct = 0.0
            nn_correct = 0.0
            nn_samples = 0.0
            for i in range(0,len(mem_df.index)):
                #Checking if memory action is in template name (ex: 'forward_high_punch' in memory and 'forward_high_punch_3' in template)
                if(vis_df['template_status'].iloc[i].find(mem_df['Action'].iloc[i]) != -1):
                    template_correct += 1.0
                #Checking if nn has a prediction, then if it's close to state memory
                if(vis_df['nn_status'].iloc[i] != 'N/A'):
                    nn_samples += 1.0
                    if(str(mem_df['Action'].iloc[i]).find(str(vis_df['nn_status'].iloc[i])) != -1):
                        nn_correct += 1
            print(c+": Overall accuracy of template matching states = %f" % (template_correct/len(mem_df.index)))
            avg_template_acc += (template_correct/len(mem_df.index))
            print(c+": Overall accuracy of neural network matching states = %f" %(nn_correct/nn_samples))
            avg_nn_acc += (nn_correct/nn_samples)

    #Find average scores
    avg_kcft_pos_acc = avg_kcft_pos_acc / (episodes*len(characters))
    avg_temp_pos_acc = avg_temp_pos_acc / (episodes*len(characters))
    avg_health_acc = avg_health_acc / (episodes*len(characters))
    avg_time_acc = avg_time_acc / (episodes*len(characters))
    avg_template_acc = avg_template_acc / (episodes*len(characters))
    avg_nn_acc = avg_nn_acc / (episodes*len(characters))
    print("\n==============Average Scores======================")
    print("Average kcft position tracking accuracy = %f" % avg_kcft_pos_acc)
    print("Average template match position tracking accuracy = %f" % avg_temp_pos_acc)
    print("Average health estimation accuracy = %f" % avg_health_acc)
    print("Average time estimation accuracy = %f" % avg_time_acc)
    print("Average template matching action prediction accuracy = %f" %avg_template_acc)
    print("Average neural network action prediction accuracy = %f" % avg_nn_acc)

    #Histogram comparing position accuracies
    fig = plt.figure()
    plt.hist(hist_kcft_pos,40,alpha=0.5,label='KCFT')
    plt.hist(hist_temp_pos,40,alpha=0.5,label='Template Match')
    plt.title("Relative Error Between Estimated and True Position (all characters)")
    plt.legend(loc='upper left')
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.show()

    #Histogram showing health accuarcy
    fig = plt.figure()
    plt.hist(hist_health,40,range=(0.0,1.2))
    plt.title("Relative Error Between Estimated and True Health (all characters)")
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.show()

    #Histogram showing health accuarcy
    fig = plt.figure()
    plt.hist(hist_time,50)
    plt.title("Relative Error Between Estimated and True Time (all characters)")
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.show()

    return



if __name__ == "__main__":
    main()
