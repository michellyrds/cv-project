import moviepy.editor as mpy

vcodec = "libx264"

videoquality = "24"

# ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
compression = "ultrafast"


def edit_video(videopath, savetitle, cuts, w=720, h=480):
    # load file
    video = mpy.VideoFileClip(videopath)

    # cut file
    clips = []
    for cut in cuts:
        clip = video.subclip(cut[0], cut[1])
        clips.append(clip)

    paths = []
    for i, clip in enumerate(clips):

        savePath = "media/temp/" + savetitle + str(i) + ".mp4"
        print(savePath)
        paths.append(savePath)
        final_clip = clip
        moviesize = w, h
        final_clip = mpy.CompositeVideoClip([final_clip], size=moviesize)

        # save file
        final_clip.write_videofile(
            savePath,
            threads=4,
            fps=24,
            codec=vcodec,
            preset=compression,
            ffmpeg_params=["-crf", videoquality],
        )

    video.close()
    return paths


if __name__ == "__main__":
    videopath = "media/input/AMD EPYC 2 Geracao.mp4"
    savetitle = "editado"
    cuts = [
        ("00:01:00.000", "00:01:20.000"),
        ("00:05:00.000", "00:05:30.000"),
        ("00:08:00.000", "00:08:15.000"),
    ]
    print(edit_video(videopath, savetitle, cuts, 1920, 1080))
