import cv2
import numpy as np
from PIL import Image



def read_video_with_timestamp(video_anntotation, num_frames=10):
    """
    Reads a video file and uniformly samples the frames.
    
    Args:
        filename (str): The filename of the video file to read.
        num_frames (int): The number of frames to sample.
    
    Returns:
        List[np.ndarray]: A list of sampled frames, where each frame is a NumPy array.
    """
    # Open the video file

    timestamps = video_anntotation["annotation"]["timestamps"]
    video_path = video_anntotation["video_path"]
    all_frames = []
    for period in timestamps:
        if period[1]-period[0]<4:
            frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=5)
        elif period[1]-period[0]<10:
            frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=8)
        elif period[1]-period[0]<30:
            frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=10)
        elif period[1]-period[0]<50:
            frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=12)
        elif period[1]-period[0]<80:
            frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=14)
        elif period[1]-period[0]<120:
            frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=16)
        else:
            frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=20)
        # elif period[1]-period[0]<150:
        #     frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=25)
        # else:
        #     frame = read_video_per_interval_sampling(video_path, period[0], period[1], num_frames=30)
        all_frames.append(frame)

    
    return all_frames





def read_video_per_interval(path, start_time, end_time, sample_interval):
    # Open the video file
    cap = cv2.VideoCapture(path)
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame number corresponding to the start and end time
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set the frame position to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize variables for sampling
    sample_frame = start_frame
    sample_time = start_time
    
    # Read the frames between the start and end frames at the specified interval
    frames = []
    while sample_frame < end_frame:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample the frame if the time has passed the sampling interval
        if sample_time >= start_time and sample_time <= end_time and sample_frame % (sample_interval * fps) == 0:
            frames.append(frame)
        
        # Increment the sample frame and time
        sample_frame += 1
        sample_time = sample_frame / fps
            
    # Release the video capture object
    cap.release()
    
    # Convert the list of frames to a numpy array
    frames = np.array(frames)
    
    return frames

def read_video_per_interval_sampling(path, start_time, end_time, num_frames=3):
    # Open the video file
    cap = cv2.VideoCapture(path)
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame number corresponding to the start and end time
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set the frame position to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize variables for sampling


    # sample_interval = (end_time - start_time) / num_frames

    num_total_frames = end_frame - start_frame

    step_size = num_total_frames // num_frames

    # print(start_frame,end_frame, num_frames, step_size)
    
    # Initialize a list to store the sampled frames
    sampled_frames = []
    
    # Loop over the frames and sample every `step_size`th frame
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if ret and (i-start_frame)% step_size ==0:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
            frame = Image.fromarray(frame)
            sampled_frames.append(frame)


    # Release the video capture object
    cap.release()
    
    # Convert the list of frames to a numpy array
    # frames = np.array(frames)

    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # frame = Image.fromarray(frame)
    
    return sampled_frames









def read_video_with_timestamp_key_frame(video_anntotation, num_frames=10):
    """
    Reads a video file and uniformly samples the frames.
    
    Args:
        filename (str): The filename of the video file to read.
        num_frames (int): The number of frames to sample.
    
    Returns:
        List[np.ndarray]: A list of sampled frames, where each frame is a NumPy array.
    """
    # Open the video file

    timestamps = video_anntotation["annotation"]["timestamps"]
    video_path = video_anntotation["video_path"]
    all_frames = []
    for period in timestamps:
        frames = key_frame_reader(video_path,period[0], period[1], num_key_frames=num_frames)
        all_frames.append(frames)

    return all_frames








def key_frame_reader(filename, start_time, end_time, num_key_frames=10):


    # num_key_frames = 10


    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    sample_rate = 0.1 # seconds
    sample_interval = int(sample_rate * cap.get(cv2.CAP_PROP_FPS))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = None
    diff_frames = []
    saved_frames = []

    start_frame = int(start_time * cap.get(cv2.CAP_PROP_FPS))
    end_frame = int(end_time * cap.get(cv2.CAP_PROP_FPS))

    # print(sample_interval)

    for i in range(total_frames):
        if i>= start_frame and i<=end_frame and i%sample_interval==0 :
            ret, frame = cap.read()
            if ret:
                if prev_frame is not None:
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_frames.append(diff)
                    saved_frames.append(frame)
                prev_frame = frame
            else:
                break

    motion_scores = []
    for frame in diff_frames:
        motion_scores.append(frame.sum())

    # print(motion_scores)
    
    key_frames = []
    
    for i in range(num_key_frames):
        


        max_index = motion_scores.index(max(motion_scores))

        key_frames.append(max_index)
        motion_scores[max_index]=-1000
    
    key_frames.sort()
    # print(motion_scores)
    # print(key_frames)
    final_frames = []

    for i, frame_index in enumerate(key_frames):

        ret, frame = cap.read()
        if ret:
            selected_frame = saved_frames[frame_index]
            selected_frame = cv2.cvtColor(selected_frame,cv2.COLOR_BGR2RGB)
            selected_frame = Image.fromarray(selected_frame)
            final_frames.append(selected_frame)
        else:
            break

        

    cap.release()

    return final_frames




def read_video_for_qa(filename):
    """
    Reads a video file and uniformly samples the frames.
    
    Args:
        filename (str): The filename of the video file to read.
        num_frames (int): The number of frames to sample.
    
    Returns:
        List[np.ndarray]: A list of sampled frames, where each frame is a NumPy array.
    """
    # Open the video file

    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    
    
    
    # Get the total number of frames in the video
    num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    period = num_total_frames/fps

    if period <4:
        num_frames=5
    elif period <10:
        num_frames=5
    elif period <20:
        num_frames=10
    elif period<30:
        num_frames=15
    elif period<40:
        num_frames=20
    elif period<50:
        num_frames=25
    elif period<60:
        num_frames=30
    elif period<70:
        num_frames=35
    else:
        num_frames=60

    # Calculate the step size for sampling frames
    step_size = num_total_frames // num_frames
    
    # Initialize a list to store the sampled frames
    sampled_frames = []
    
    # Loop over the frames and sample every `step_size`th frame
    for i in range(num_total_frames):
        ret, frame = cap.read()
        if ret and i % step_size == 0:
            # Convert the frame to grayscale
#             frame = cv2.cvtColor(frame).astype(np.float32)
            # Append the frame to the list of sampled frames
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
            frame = Image.fromarray(frame)
            sampled_frames.append(frame)
    
    # Release the video capture object
    cap.release()
    
    return sampled_frames

def read_video_sampling(filename, num_frames):

    """
    Reads a video file and uniformly samples the frames.
    
    Args:
        filename (str): The filename of the video file to read.
        num_frames (int): The number of frames to sample.
    
    Returns:
        List[np.ndarray]: A list of sampled frames, where each frame is a NumPy array.
    """
    # Open the video file

    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    
    
    # Get the total number of frames in the video
    num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    # Calculate the step size for sampling frames
    step_size = num_total_frames // num_frames
    
    # Initialize a list to store the sampled frames
    sampled_frames = []
    
    # Loop over the frames and sample every `step_size`th frame
    for i in range(num_total_frames):
        ret, frame = cap.read()
        if ret and i % step_size == 0:
            # Convert the frame to grayscale
#             frame = cv2.cvtColor(frame).astype(np.float32)
            # Append the frame to the list of sampled frames
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
            frame = Image.fromarray(frame)
            sampled_frames.append(frame)
    
    # Release the video capture object
    cap.release()
    
    return sampled_frames