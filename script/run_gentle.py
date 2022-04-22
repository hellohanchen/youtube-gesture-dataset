# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import glob
import logging
import multiprocessing
import os
import re
import sys
import subprocess

from tqdm import tqdm

from config import *
from make_ted_dataset import read_subtitle
from resample import resample, resampled
from config import my_config

sys.path.append('/mnt/Samsung860/han/csci699/gentle')
import gentle


OUTPUT_TRANSCRIPT_PATH = my_config.WORK_PATH + "/temp_transcript"
OUTPUT_WAV_PATH = my_config.WORK_PATH + "/temp_wav"

# prepare gentle
nthreads = multiprocessing.cpu_count() - 2
logging.getLogger().setLevel("WARNING")
disfluencies = set(['uh', 'um'])
resources = gentle.Resources()


def run_gentle(video_path, vid, result_path):
    vtt_subtitle = read_subtitle(vid)
    transcript = ''
    for i, sub in enumerate(vtt_subtitle):
        transcript += (vtt_subtitle[i].text + ' ')
    transcript = re.sub('\n', ' ', transcript)  # remove newline characters

    if transcript == "":
        print("ERROR: Empty transcript, skip [" + vid + "]")
        return

    transcript_path = OUTPUT_TRANSCRIPT_PATH + '/' + vid + '_transcript.txt'
    with open(transcript_path, 'w') as tfile:
        tfile.write(transcript)

    # align
    resampled_file = OUTPUT_WAV_PATH + '/' + vid + '_resampled.wav'
    resample(video_path, resampled_file)

    if os.path.exists(result_path):
        print("Aligned result exists, skip [" + vid + "]")
        return

    # with resampled(video_path) as wav_file:
        # aligner = gentle.ForcedAligner(resources, transcript, nthreads=nthreads, disfluency=False, conservative=False,
        #                                disfluencies=disfluencies)
        # result = aligner.transcribe(wav_file, logging=logging)

        # run gentle with docker running on port 8765
    command = "curl -F \"audio=@"+ resampled_file + "\" -F \"transcript=@" \
              + transcript_path + "\" \"http://localhost:8765/transcriptions?async=false\" > " + result_path
    print(command)
    subprocess.call(command, shell=True)

    # write results
    # with open(result_path, 'w', encoding="utf-8") as fh:
    #     fh.write(result.to_json(indent=2))


def main():
    videos = glob.glob(my_config.VIDEO_PATH + "/*.mp4")
    n_total = len(videos)
    for i, file_path in tqdm(enumerate(sorted(videos, key=os.path.getmtime))):
        vid = os.path.split(file_path)[1][-15:-4]
        print('{}/{} - {}'.format(i+1, n_total, vid))
        result_path = my_config.VIDEO_PATH + '/' + vid + '_align_results.json'
        if os.path.exists(result_path) and os.path.getsize(result_path):  # existing and not empty
            print('JSON file already exists ({})'.format(vid))
        else:
            run_gentle(file_path, vid, result_path)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_TRANSCRIPT_PATH):
        os.makedirs(OUTPUT_TRANSCRIPT_PATH)
    if not os.path.exists(OUTPUT_WAV_PATH):
        os.makedirs(OUTPUT_WAV_PATH)

    main()
