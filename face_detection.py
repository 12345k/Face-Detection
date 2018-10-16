import face_recognition
from PIL import Image, ImageDraw
import os
import re
import scipy.misc
import warnings
import face_recognition.api as face_recognition
import sys
import cv2
import numpy as np

known_people_folder ="/home/desktop-su-02/Documents/face-recongition-master/known_image"
image_to_check="/home/desktop-su-02/Documents/face-recongition-master/live_feed"

known_face_encodings=[]
known_face_names=[]

for known_people_image in os.listdir(known_people_folder):
        basename = known_people_image 
        img = face_recognition.load_image_file(os.path.join(known_people_folder,known_people_image))
        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 1:
            known_face_names.append(basename)
            known_face_encodings.append(encodings[0])   


for to_check in os.listdir(image_to_check):  
	
	unknown_image = face_recognition.load_image_file(to_check)

	
	face_locations = face_recognition.face_locations(unknown_image)
	face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

	pil_image = Image.fromarray(unknown_image)
	
	draw = ImageDraw.Draw(pil_image)

	
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
	    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

	    name = "Unknown"

	    if True in matches:
	        first_match_index = matches.index(True)
	        name = known_face_names[first_match_index]

	    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

	    text_width, text_height = draw.textsize(name)
	    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
	    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


	
	del draw

	

	pil_image.show()
	# cv2.imshow("Image",pil_image)
	

	# You can also save a copy of the new image to disk if you want by uncommenting this line
	# pil_image.save("image_with_boxes.jpg")