import cv2

#The video
#video = cv2.VideoCapture('Tesla Dashcam Accident.mp4')
video = cv2.VideoCapture('Cars And People.mp4')

#our pre-trained car/human classifier
car_tracker_file = 'car_detector.xml'
human_tracker_file = 'human_detector.xml'

#creat car/human classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
human_tracker = cv2.CascadeClassifier(human_tracker_file)

#run forever until car stops or somethings
while True:

    #read the current frame
    (read_successful, frame) = video.read()

    #safe coding
    if read_successful:
        #must covert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and humans
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    humans = human_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangeles around the cars
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0),1)
    
    #draw rectangles around humans
    for(x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255) , 1)
        
    #Display the image with cars/faces spotted
    cv2.imshow('Franks Car/Human Detector', frame) 

    #dont autoclose(wait here in code and listen for key press)
    key = cv2.waitKey(1)

    #stop if Q key is pressed
    if key==81 or key ==113:
        break

#release the videocapture object
video.release()


print("Code Completed")
