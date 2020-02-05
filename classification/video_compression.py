import os
import subprocess
import argparse
from os.path import join
import detect_from_video

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_dir', '-i', type=str)
    p.add_argument('--compression_factor', '-c', type=int, default = 24)
    
    # optional arguments if detection is true
    p.add_argument('--model_dir', '-mdir', type=str, 
        default="/data2/paarth/faceforensics++_models_subset/face_detection/") #dir contains xception, meso
    p.add_argument('--model_type', '-mtype', type=str, 
        default="xception")

    p.add_argument('--detect', action='store_true')
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    if args.input_dir[-1] == '/':
        args.input_dir = args.input_dir[:-1]
        
    videos = os.listdir(args.input_dir)

    videos = [ video for video in videos if (video.endswith(".mp4") or video.endswith(".avi")) ]
    args.output_dir = args.input_dir + "_" + str(args.compression_factor)

    os.makedirs(args.output_dir, exist_ok=True)

    for video in videos:
        print ("Compressing", video)
        subprocess.call(['ffmpeg', '-y', '-i', join(args.input_dir, video),  '-vcodec', 
            'libx264', '-crf', '{}'.format(args.compression_factor), join(args.output_dir, video.replace('.avi', '.mp4'))])

    if args.detect:
        model_type = args.model_type
        if model_type == "xception":
            model_path = join(args.model_dir, "xception", "all_{}.p".format("c23"))
        elif model_type == "meso":
            model_path = join(args.model_dir, "Meso", "Meso4_deepfake.pkl")        

        detected_folder_path = args.output_dir + "_detected"
        os.makedirs(detected_folder_path, exist_ok=True)
        for video in videos:
            adv_video_path = join(args.output_dir, video.replace(".avi", ".mp4"))
            detect_from_video.test_full_image_network(
                video_path = adv_video_path,
                model_path = model_path,
                model_type = model_type,
                output_path = detected_folder_path,
                start_frame = 0,
                end_frame = None,
                cuda = args.cuda
            )


if __name__ == '__main__':
    main()