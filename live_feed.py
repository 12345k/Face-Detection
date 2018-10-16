# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import re
import scipy.misc
import warnings
import face_recognition.api as face_recognition
import sys
import cv2


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for known_people_image in os.listdir(known_people_folder):
        basename = known_people_image 
        img = face_recognition.load_image_file(os.path.join(known_people_folder,known_people_image))
        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 1:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])   
    
    return known_names, known_face_encodings


def test_image(image_to_check, known_names, known_face_encodings):

    distance_calc=[]
    result_image=[]
    for to_check in os.listdir(image_to_check): 
        unknown_image = face_recognition.load_image_file(to_check)

        # Scale down image if it's giant so things run a little faster
        if unknown_image.shape[1] > 1600:
            scale_factor = 1600.0 / unknown_image.shape[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                unknown_image = scipy.misc.imresize(unknown_image, scale_factor)

        unknown_encodings = face_recognition.face_encodings(unknown_image)
        

        if len(unknown_encodings)==1:
            for unknown_encoding in unknown_encodings:
                result = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
                distance = face_recognition.face_distance(known_face_encodings, unknown_encoding)
                print(distance[0])
                print("True") if True in result else print("False ")

            distance_calc.append (distance[0])
            result_image.append(result[0])
        else:
            distance_calc.append (0)
            result_image.append("Many Faces or No Faces")
    return distance_calc,result_image



def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def main(known_people_folder, image_to_check):
    known_names, known_face_encodings = scan_known_people(known_people_folder)
    distance,result=test_image(image_to_check, known_names, known_face_encodings)
    return distance,result
    


main("/home/desktop-su-02/Documents/face-recongition-master/known_image","/home/desktop-su-02/Documents/face-recongition-master/live_feed")




