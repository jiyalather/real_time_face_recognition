import cv2
import os
import pickle
import face_recognition
import cvzone
import numpy as np  # ✅ Add this if missing

# Try different camera indexes (0, 1, 2)
for i in range(-1, 3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} opened successfully")
        break  # Stop at the first working camera

# If no camera was found, exit
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(3, 498)  # Set width
cap.set(4, 298)   # Set height


# Ensure the correct path and file type
imgBackground = cv2.imread('Resources/background.png')  # Check the extension


# Verify if image is loaded
if imgBackground is None:
    print("Error: Could not load the background image. Check file path and format.")
    exit()
    

folderModePath ='Resources/Modes'
modePathList = os.listdir(folderModePath)

imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
    
#print(len(imgModeList))

#load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown,studentIds = encodeListKnownWithIds
print("Encode File Loaded")



while True:
    success, img = cap.read()

    if not success:  # Handle camera read error
        print("Error: Failed to capture image.")
        break

    imgS= cv2.resize(img,(0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    
    img_resized = cv2.resize(img, (498, 298))
    imgBackground[150:150+298,35:35+498]=img_resized
    target_size = (226,350)  # (Width, Height) → swap the numbers based on your background
    imgModeList[2] = cv2.resize(imgModeList[2], target_size)
    imgBackground[44:44+350,700:700+226]=imgModeList[2]

    for encodeFace, faceLoc in zip(encodeCurFrame , faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    #     print("matches",matches)
    #     print("facesDis", faceDis)


    matchIndex = np.argmin(faceDis)
    # print("Match Index", matchIndex)

    if matches[matchIndex]:
       # print("Known Face Detected")
       # print(studentIds[matchIndex])
       y1, x2, y2, x1 = faceLoc
       y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
       bbox = 55+x1, 162+y1, x2-x1, y2-y1 
       imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)



    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
     matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    
    best_match_index = None
    if len(faceDis) > 0:  # Ensure there are known faces to compare
        best_match_index = faceDis.argmin()  # Get index of the best match
    
    if best_match_index is not None and matches[best_match_index]:  # If a match is found
        student_id = studentIds[best_match_index]  # Get the ID of the matched person
        print(f"Face recognized: {student_id}")

        # Display the recognized name near the face
        top, right, bottom, left = faceLoc
        top, right, bottom, left = top*4, right*4, bottom*4, left*4  # Resize back

        cv2.rectangle(imgBackground, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(imgBackground, student_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




    #cv2.imshow("webcam", img)
    cv2.imshow("Face Attendance", imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()