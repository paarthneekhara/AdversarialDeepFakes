import os
from os.path import join
import argparse
import copy
import json
import pprint
import time

def main():
    """
    Aggregates attack statistics to the experiments running.
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--exp_folder', '-exp', type=str, 
        default="/data2/DFExperiments") # where sub directories will be created
    
    args = p.parse_args()
    experiment_dir = args.exp_folder
    
    model_types = ["xception", "meso"]
    fake_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    compression_types = ["c23", "c40", "raw"]
    attack_methods = ["robust", "iterative_fgsm", "carlini_wagner", "black_box", "black_box_robust"]


    experiment_stats = {}

    while True:
        # sleep for minute
        for model_type in model_types:
            experiment_stats[model_type] = {}
            for fake_type in fake_types:
                experiment_stats[model_type][fake_type] = {}
                for compression_type in compression_types:
                    experiment_stats[model_type][fake_type][compression_type] = {}
                    for attack_method in attack_methods:
                        
                        for extension in ["", "_compressed", "_23", "_30", "_40"]:
                            experiment_stats[model_type][fake_type][compression_type]["{}{}".format(attack_method, extension)] = {}
                            adv_videos_dir = join(experiment_dir, model_type, fake_type, compression_type, "adv_{}{}".format(attack_method, extension))
                            if extension in ["_23", "_30", "_40"]:
                                detected_videos_dir = join(experiment_dir, model_type, fake_type, compression_type, "adv_{}{}_detected".format(attack_method, extension))    
                            else:
                                detected_videos_dir = join(experiment_dir, model_type, fake_type, compression_type, "adv_{}_detected{}".format(attack_method, extension))
                            
                            if os.path.isdir(adv_videos_dir) and os.path.isdir(detected_videos_dir):
                                attack_json_files = os.listdir(adv_videos_dir)
                                attack_json_files = [file for file in attack_json_files if file.endswith("metrics_attack.json")]

                                detect_json_files = os.listdir(detected_videos_dir)
                                detect_json_files = [file for file in detect_json_files if file.endswith("metrics.json")]

                                attack_metrics = {
                                    'avg_frame_fooling_rate' : None,
                                    'avg_video_fooling_rate' : None,
                                    'total_videos_fooled' : 0,
                                    'total_frames_fooled' : 0,
                                    'avg_linf_norm' : None,
                                    'total_videos' : 0,
                                    'total_frames' : 0
                                }


                                for detect_file in detect_json_files:
                                    with open(join(detected_videos_dir, detect_file)) as f:
                                        try:
                                            _attack_metrics = json.loads(f.read())
                                        except:
                                            print ("Error loading", detect_file)
                                            continue
                                        attack_metrics['total_frames'] += _attack_metrics['total_frames']

                                        if _attack_metrics['percent_fake_frames'] < 0.5:
                                            attack_metrics['total_videos_fooled'] += 1.

                                        attack_metrics['total_videos'] += 1.
                                        attack_metrics['total_frames_fooled'] += _attack_metrics['total_real_frames']

                                if attack_metrics['total_videos'] > 0 and attack_metrics['total_frames'] > 0:
                                    attack_metrics['avg_video_fooling_rate'] = attack_metrics['total_videos_fooled']/attack_metrics['total_videos']
                                    attack_metrics['avg_frame_fooling_rate'] = attack_metrics['total_frames_fooled']/attack_metrics['total_frames']

                                total_l_infnorm = 0.0
                                total_frames = 0.0
                                for attack_file in attack_json_files:
                                    with open(join(adv_videos_dir, attack_file)) as f:
                                        try:
                                            _attack_metrics = json.loads(f.read())
                                        except:
                                            continue
                                        for frame_meta_data in _attack_metrics['attack_meta_data']:
                                            total_l_infnorm += frame_meta_data['l_inf_norm']
                                            total_frames += 1.

                                if total_frames > 0:
                                    attack_metrics['avg_linf_norm'] = total_l_infnorm/total_frames


                                experiment_stats[model_type][fake_type][compression_type]["{}{}".format(attack_method, extension)] = copy.copy(attack_metrics)
                                with open(join(experiment_dir, "agg_exp_stats.json"), "w") as wf:
                                    wf.write(json.dumps(experiment_stats))

        pprint.pprint(experiment_stats)
        time.sleep(10)



if __name__ == '__main__':
    main()
