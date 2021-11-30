import cv2 as cv
import os
import matplotlib.pyplot as plt
import MTM

def loadTemplates(fileloc):
	'''
	Loads all images from folder into a list

	Inputs: 
		fileloc: Directory of template images

	Returns:
		templates: [templateName, templateImage]
	'''
	templates = []
	for filename in os.listdir(fileloc):
		template_img = cv.cvtColor(cv.imread(os.path.join(fileloc, filename)),cv.COLOR_BGR2RGB)
		template_img = cv.cvtColor(template_img, cv.COLOR_RGB2GRAY)
		templates.append((filename.split('.')[0], template_img))
	return templates


def getMatchLocation(image,templates):
	'''
	Looks for a match and returns the location and label

	Inputs: 
		image: Image to check for matches
		templates: List of labeled template images

	Returns:
		label: matchLabel
		location: (X,Y,Width,Height)
	'''
	image_bw = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
	match = MTM.matchTemplates(templates,
							  image_bw,
							  score_threshold=0.3,
							  maxOverlap = 0,
							  searchBox=(0, 60, 250, 200),
							  N_object=1,
							  method=cv.TM_CCOEFF_NORMED)
	label = match['TemplateName'].iloc[0]
	location = match['BBox'].iloc[0]
	return (label,location)

def drawBoundingBox(image,location):
	'''
	Draws a bounding box around the detected match

	Inputs: 
		image: Image to overlay
		location: (X,Y,Width, Height)

	Returns:
		overlay: Image overlayed with the bounding box
	'''
	overlay = cv.rectangle(image,(location[0],location[1]),(location[0]+location[2],location[1]+location[3]),(255,0,0),2)
	plt.imshow(overlay)
	plt.show(block=False)
	plt.pause(0.0001)
	return (overlay)
