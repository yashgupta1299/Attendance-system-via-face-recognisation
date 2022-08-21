# A Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

from tkinter.ttk import Treeview
import cv2
import numpy as np

# Initialise Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
file_name = input("Enter the name of a person : ")
while True:
	ret,frame = cap.read()

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	

    # here faces are list of tuples acc to number of faces preset
    # (x,y,w,h)
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue

    # sorting the faces accordind to its area x,y,w,h
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	
	# for face in faces[-1:]:
	face=faces[-1]
	x,y,w,h = face
	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)

	#Extract (Crop out the required face) : Region of Interest
	offset = 10

	# in frame first is y and second is x
	face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
	
	try:
		# changing size of image to 100 by 100
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
	except Exception as e:
		pass


	try:
		cv2.imshow("Frame",frame)
		cv2.imshow("Face Section",face_section)
	except Exception as e:
		pass

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0],-1))
# if we want to reduce size of file we can use grayframe 
# then instead of three we get 1 dimension only
print(face_data.shape)



# Save this data into file system
dataset_path = './data/'
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()
