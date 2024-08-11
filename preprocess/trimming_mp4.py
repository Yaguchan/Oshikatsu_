from moviepy.video.io.VideoFileClip import VideoFileClip


# python preprocess/trimming_mp4.py
INPATH = './data/62thSingleDance_1k.mp4'
START = 178
END = 182


def main():
    video = VideoFileClip(INPATH)
    trimmed_video = video.subclip(START, END)
    # OUTPATH = INPATH.replace('full_1k.mp4', f'{START}_{END}_1k.mp4')
    OUTPATH = INPATH.replace('1k.mp4', f'{START}_{END}_1k.mp4')
    trimmed_video.write_videofile(OUTPATH, codec="libx264")


if __name__ == '__main__':
    main()