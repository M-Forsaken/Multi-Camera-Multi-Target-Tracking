import cv2
from ctypes import windll, c_short
windll.shcore.SetProcessDpiAwareness(1)


# Create an object to read
# from camera
cam_1_url = "http://192.168.0.110:4747/video"
cam_2_url = "http://192.168.0.121:4747/video"
cam_3_url = "http://192.168.0.103:4747/video"
video_1 = cv2.VideoCapture(cam_1_url)
video_2 = cv2.VideoCapture(cam_2_url)
video_3 = cv2.VideoCapture(cam_3_url)

# We need to check if camera
# is opened previously or not
if (video_1.isOpened() == False):
    print("Error reading video file")


size = (640, 400)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result_1 = cv2.VideoWriter(
    'cam_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
result_2 = cv2.VideoWriter(
    'cam_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
result_3 = cv2.VideoWriter(
    'cam_3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

while (True):
    ret, frame = video_1.read()
    ret_1, frame_1 = video_2.read()
    ret_2, frame_2 = video_3.read()

    if ret == True:

        # Write the frame into the
        # file 'filename.avi'
        frame = cv2.resize(frame,size)
        frame_1 = cv2.resize(frame_1,size)
        frame_2 = cv2.resize(frame_2, size)
        result_1.write(frame)
        result_2.write(frame_1)
        result_3.write(frame_2)


        # Display the frame
        # saved in the file
        cv2.imshow('Frame', frame)
        cv2.imshow('Frame2', frame_1)
        cv2.imshow('Frame3', frame_2)

        # Press S on keyboard
        # to stop the process
        if cv2.waitKey(30) & 0xFF == ord('s'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture and video
# write objects
video_1.release()
result_1.release()
video_2.release()
result_2.release()
video_3.release()
result_3.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
