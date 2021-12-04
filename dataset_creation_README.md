#This will contain information about the project.
#Here's how I'm collecting data:

#Setup Gym-retro w/ Street Fighter II Genesis:
  #pip install gym-retro
  #python3 -m retro.import ./roms

#Setup machine learning agent
  #Added the following directory: https://github.com/corbosiny/StreetFighterAI
  #Follow the directions on the readme there to replace the original StreetFighter folder in gym-retro's data

#Collecting Data
  #The code I added for collecting data is in StreetFighterAI/src/Lobby.py
  #It's currently set up to only handle 1 'episode' at a time. Each episode has Ryu fighting a
  match against each other character
  #To collect data from the trained agent run 'python3 DeepQAgent.py -e 1'

#The Datasets
  #Each test within an episode has 2 datasets, named after the character Ryu is fighting
  #character_images.npy is a numpy array containing a series of RGB images. The first axis is the frame the image was taken at.
  #character_statespace.csv contains the relevant state space information we're trying to identify.
    #We need to do some more work here to interpret the data:
      #status is probably defined somewhere in the StreetFighterAI code. We need to interpret the integers as labels like "standing", "punch", ect
      #round_timer is counting milliseconds or something, not the time that will be displayed on screen

#Datasets Retrieval for the neural network
    #Keep the training dataset insidle the folder used in the NN_recog.py, additionally, the dataset is randomized for learning.
    #Total of 683 images for 7 classes of actions are used.
