import retro
import keyboard as kb
import time
import cv2 as cv
import templateMatching as tm


def keyboardToAction():
	KICK_LOW = 0
	KICK_MEDIUM = 0
	KICK_HIGH = 0
	JUMP = 0
	CROUCH = 0
	LEFT = 0
	RIGHT = 0
	PUNCH_LOW = 0
	PUNCH_MEDIUM = 0
	PUNCH_HIGH = 0

	if(kb.is_pressed("d")):
		RIGHT = 1
	if(kb.is_pressed("a")):
		LEFT = 1
	if(kb.is_pressed("s")):
		CROUCH = 1
	if(kb.is_pressed("w")):
		JUMP = 1
	if(kb.is_pressed("space")):
		PUNCH_HIGH = 1
	return [KICK_LOW, KICK_MEDIUM, 0, 0, JUMP, CROUCH, LEFT, RIGHT, KICK_HIGH, PUNCH_LOW, PUNCH_MEDIUM, PUNCH_HIGH]

def main():
	env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
	obs = env.reset()
	
	action = keyboardToAction()
	obs,rew,done,info = env.step(action)

	templates = tm.loadTemplates('assets/templates')

	counter = 0
	done = False
	while(not done):
		test_img = env.render(mode='rgb_array')
		match_label,match_loc = tm.getMatchLocation(test_img,templates)
		#tm.drawBoundingBox(test_img,match_loc)
		
		print(match_label,match_loc)
		env.render()
		
		action = keyboardToAction()
		obs,rew,done,info = env.step(action)
	env.close()

if __name__ == '__main__':
	main()