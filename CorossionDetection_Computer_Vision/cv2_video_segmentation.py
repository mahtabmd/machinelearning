import pixellib
from pixellib.instance import instance_segmentation

cap = cv2.VideoCapture('videos/Abandoned_Mill_PA.mp4')

segment_video = instance_segmentation()
segment_video.load_model("corossion_model_gabor_rf")
segment_video.process_video(cap, overlay = True, frames_per_second= 15, output_video_name="output_video.mp4")
