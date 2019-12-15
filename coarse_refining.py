from sklearn.metrics import mutual_info_score
import cv2
import numpy as np
detector = cv2.ORB_create()

video_number = 'v12'

video_path = video_number + '.mp4'
print ('Coarse Refining: ', video_path)
capture = cv2.VideoCapture(video_path)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


detector = cv2.ORB_create()

def coarse_refine(frame1,frame2):
    
    kp1 = detector.detect(frame1, None)
    kp1 , des1 = detector.compute(frame1, kp1)
    des1 = np.array(des1)

    kp2 = detector.detect(frame2, None)
    kp2 , des2 = detector.compute(frame2, kp2)
    des2 = np.array(des2)

    if not des1.any():
        print ('des1 empty')
        return 0
    elif not des2.any():
        print ('des2 empty')
        return 0
    else:

        #print ('frame1  = ', frame1.shape, ', shape = ', frame2.shape, ',des1 = ', des1)
        #print ('des1 type = ', type(des1), ', shape = ', des1.shape)
        #print ('=', des1)
        r1,c1 = des1.shape
        #print ('des1 rows = ', r1 , ', des1 cols = ', c1)
        des1 = np.reshape(des1, (r1*c1))
        des1 = des1[0:(r1*c1)]
        #print ('des1 shape = ', des1.shape)

        
        r2,c2 = des2.shape
        #print ('des2 rows = ', r2 , ', des2 cols = ', c2)
        des2 = np.reshape(des2, (r2*c2))
        #print ('des2 shape = **', des2.shape)

        shape_des1 = r1*c1
        shape_des2 = r2*c2

        if shape_des1 > shape_des2:
            des1 = des1[0:(r2*c2)]
            des2 = des2[0:(r2*c2)]
        elif shape_des2 > shape_des1:
            des1 = des1[0:(r1*c1)]
            des2 = des2[0:(r1*c1)]
        else:
            des1 = des1[0:(r1*c1)]
            des2 = des2[0:(r2*c2)]


        #print ('des2 shape = ', des2.shape)
        mi = mutual_info_score(des1,des2)
        
        return mi


def video_processing():
    frames_counter = 0

    while(frames_counter < total_frames-1):

        frames_counter = frames_counter + 1
        print ('Processing ', str(frames_counter), ' out of ', total_frames)
        ret, frame = capture.read()
        cv2.imshow('f', frame)
        cv2.waitKey(5)
        if (ret):
            previous_frame = frame
            status, frame = capture.read()
            frames_distance = coarse_refine(frame,previous_frame)
            if frames_distance > 1.8:
                name = video_number + '\\Coarse-refine\\'+str(frames_counter)+'.jpg'
                cv2.imwrite(name,frame)
                print ('Frame written ', name, ',distance = ',frames_distance)
            print ('m i = ' , frames_distance)

video_processing()