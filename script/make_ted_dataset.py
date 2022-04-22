# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import os
import pickle
import math
import librosa
import numpy as np

from tqdm import tqdm_gui
import unicodedata

from data_utils import *


AUDIO_DIR = my_config.WORK_PATH + "/temp_wav/"
POSE_DIR = my_config.WORK_PATH + "/pose_3d/"
DATASET = "all_" + my_config.DATASET


def read_subtitle(vid):
    postfix_in_filename = '-en.vtt'
    file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)
    if len(file_list) == 0:
        print("No -en.vtt for " + vid + " try with -en-auto.vtt")
        postfix_in_filename = '-en-auto.vtt'
        file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)

    if len(file_list) > 1:
        print('more than one subtitle. check this.', file_list)
        assert False
    if len(file_list) == 1:
        return WebVTT().read(file_list[0])
    else:
        return []


# turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def normalize_subtitle(vtt_subtitle):
    for i, sub in enumerate(vtt_subtitle):
        vtt_subtitle[i].text = normalize_string(vtt_subtitle[i].text)
    return vtt_subtitle


def extract_audio_feature(y, sr, video_total_frames, video_start_frame, video_end_frame):
    # roi
    n = y.shape[0]
    start_frame = math.floor(video_start_frame / video_total_frames * n)
    end_frame = math.ceil(video_end_frame / video_total_frames * n)
    y_roi = y[start_frame:end_frame]

    # feature extraction
    melspec = librosa.feature.melspectrogram(y=y_roi, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels * time

    # change type
    log_melspec = log_melspec.astype('float16')
    y_roi = y_roi.astype('float16')

    return log_melspec, y_roi


def make_ted_gesture_dataset():
    dataset_train = []
    dataset_val = []
    dataset_test = []
    n_saved_clips = [0, 0, 0]

    video_files = sorted(glob.glob(my_config.VIDEO_PATH + "/*.mp4"), key=os.path.getmtime)
    for v_i, video_file in enumerate(tqdm_gui(video_files)):
        vid = os.path.split(video_file)[1][-15:-4]
        print(vid)

        # load clip, video, and subtitle
        clip_data = load_clip_data(vid)
        if clip_data is None:
            print('[ERROR] clip data file does not exist!')
            break

        video_wrapper = read_video(my_config.VIDEO_PATH, vid)
        video_total_frame = video_wrapper.total_frames

        subtitle_type = my_config.SUBTITLE_TYPE
        subtitle = SubtitleWrapper(vid, subtitle_type).get()

        if subtitle is None:
            print('[WARNING] subtitle does not exist! skipping this video.')
            continue

        # load audio
        y, sr = librosa.load(AUDIO_DIR + vid + '_resampled.wav', mono=True, sr=16000, res_type='kaiser_fast')

        # load 3d pose
        pose_data = np.load(POSE_DIR + vid + '_pose.npy')

        dataset_train.append({'vid': vid, 'clips': []})
        dataset_val.append({'vid': vid, 'clips': []})
        dataset_test.append({'vid': vid, 'clips': []})

        word_index = 0
        valid_clip_count = 0
        for ia, clip in enumerate(clip_data):
            start_frame_no, end_frame_no, clip_pose_all = clip['clip_info'][0], clip['clip_info'][1], clip['frames']
            start_time = video_wrapper.frame2second(start_frame_no)
            end_time = video_wrapper.frame2second(end_frame_no)

            # audio
            clip_audio_feature, clip_audio_raw = extract_audio_feature(y, sr, video_total_frame, start_frame_no, end_frame_no)

            clip_word_list = []

            # skip FALSE clips
            if not clip['clip_info'][2]:
                continue

            # train/val/test split
            if valid_clip_count % 10 == 9:
                dataset = dataset_test
                dataset_idx = 2
            elif valid_clip_count % 10 == 8:
                dataset = dataset_val
                dataset_idx = 1
            else:
                dataset = dataset_train
                dataset_idx = 0
            valid_clip_count += 1

            # get subtitle that fits clip
            for ib in range(word_index - 1, len(subtitle)):
                if ib < 0:
                    continue

                word_start_time = video_wrapper.second2frame(subtitle[ib]['start'])
                word_end_time = video_wrapper.second2frame(subtitle[ib]['end'])
                word_start_frame = video_wrapper.second2frame(word_start_time)
                word_end_frame = video_wrapper.second2frame(word_end_time)
                word = subtitle[ib]['word']

                if word_start_frame >= end_frame_no:
                    word_index = ib
                    break

                if word_end_frame <= start_frame_no:
                    continue

                word = normalize_string(word)
                clip_word_list.append([word, word_start_time, word_end_time, word_start_frame, word_end_frame])

            if clip_word_list:
                clip_skeleton = []

                # get skeletons of the upper body in the clip
                for frame in clip_pose_all:
                    if frame:
                        clip_skeleton.append(get_skeleton_from_frame(frame)[:24])
                    else:  # frame with no skeleton
                        clip_skeleton.append([0] * 24)

                # proceed if skeleton list is not empty
                if len(clip_skeleton) > 0:
                    # save subtitles and skeletons corresponding to clips
                    n_saved_clips[dataset_idx] += 1
                    dataset[-1]['clips'].append({'words': clip_word_list,
                                                 'skeletons': clip_skeleton,
                                                 'skeletons_3d': pose_data[start_frame_no:end_frame_no,-10:],
                                                 'start_time': start_time, 'end_time': end_time,
                                                 'start_frame_no': start_frame_no, 'end_frame_no': end_frame_no,
                                                 'audio_feature': clip_audio_feature, 'audio_raw': clip_audio_raw,
                                                 'vid': vid
                                                 })
                    print('{} ({}, {})'.format(vid, start_frame_no, end_frame_no))
                else:
                    print('{} ({}, {}) - consecutive missing frames'.format(vid, start_frame_no, end_frame_no))

    # for debugging
    # if vid == 'yq3TQoMjXTw':
    #     break

    print('writing to pickle...')
    with open(DATASET + '_train.pickle', 'wb') as f:
        pickle.dump(dataset_train, f)
    with open(DATASET + '_train_small.pickle', 'wb') as f:  # for debugging
        pickle.dump(dataset_train[0:10], f)
    with open(DATASET + '_val.pickle', 'wb') as f:
        pickle.dump(dataset_val, f)
    with open(DATASET + '_test.pickle', 'wb') as f:
        pickle.dump(dataset_test, f)

    print('no. of saved clips: train {}, val {}, test {}'.format(n_saved_clips[0], n_saved_clips[1], n_saved_clips[2]))


if __name__ == '__main__':
    make_ted_gesture_dataset()
