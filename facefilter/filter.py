import face_recognition
import cv2
import numpy
from gtts import gTTS
import pygame
pygame.init()
import datetime
import time

def speak(text): 
	x = "Good morning " + text + " !"		# 

	tts = gTTS(text=x,lang='en')			# Text-to-speech
	tts.save('temp.mp3')					# save the speech to a temporary mp3 file

	print("Data: ",x)						# print the data in the terminal
	pygame.mixer.music.load('temp.mp3')		#
	pygame.mixer.music.play()				# Pygame music player	
	time.sleep(3)						#
	pygame.mixer.music.fadeout(5)			#	
	
sean_image = face_recognition.load_image_file("sean.jpg")
sean_face_encoding = face_recognition.face_encodings(sean_image)[0]

rowel_image = face_recognition.load_image_file("rowel.jpg")
rowel_face_encoding = face_recognition.face_encodings(rowel_image)[0]

bibay_image = face_recognition.load_image_file("bibay.jpg")
bibay_face_encoding = face_recognition.face_encodings(bibay_image)[0]

known_face_encodings = [bibay_face_encoding,sean_face_encoding,rowel_face_encoding]
known_face_names = ["Bibay","Sean","Rowel"]

def identify(frame):
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	rgb_small_frame = small_frame[:, :, ::-1]

	face_locations = []
	face_encodings = []
	face_names = []
	process_this_frame = True

	face_locations = face_recognition.face_locations(rgb_small_frame)
	face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)    

	face_names = []
	for face_encoding in face_encodings:
		# See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
		name = "Unknown"

	    # If a match was found in known_face_encodings, just use the first one.
		if True in matches:
			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]

		face_names.append(name)
	
	speak(name)
	return

vc = cv2.VideoCapture(0)
ret, frame = vc.read()
identify(frame)	

while True:	
	specs = cv2.imread("nice2.png",-1)
	ret, frame = vc.read()
	#identify(frame)
	frame = cv2.resize(frame, (0,0) , fx = 0.4, fy = 0.4)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	eyes_cascade = cv2.CascadeClassifier('frontalface.xml')
	
	eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
	
	for (x,y,w,h) in eyes:
		cv2.rectangle(gray, (x,y), (x+w, y+h), (255,86,30), 3)
		break

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
	specs = cv2.resize(specs, (w,h))	

	w, h, c = specs.shape

	for i in range(0, w):
	    for j in range(0, h):
	        if specs[i, j][3] != 0:
	            frame[y + i+6, x + j] = specs[i, j]

	cv2.imshow('Vid',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		#cv2.imwrite('scale.jpg',frame)		
		break

vc.release()
cv2.destroyAllWindows()
