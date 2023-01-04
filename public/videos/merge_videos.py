# by Pavlo Bazilinskyy <p.bazilinskyy@tue.nl>
import os
for x in range(0, 124):
    f1 = 'video_' + str(x) + '.mp4'
    f1_compressed = 'video_compressed_' + str(x) + '.mp4'
    if x < 64:  # groups 0 and 1
        f2 = 'video3rd_' + str(x) + '.mp4'
    else:  # group 2
        f2 = 'video3rd_' + str(x-60) + '.mp4'
    f2_compressed = 'video3rd_compressed_' + str(x) + '.mp4'
    out = 'video_merged_' + str(x) + '.mp4'  # file with merged video
    w = 640  # width of video
    h = 360  # height of video
    # compress and resize video 1
    cmd = "ffmpeg -y -i " + f1 + " -s " + str(w) + "x" + str(h) + " -vcodec h264 -b:v 1000k -acodec mp2 " + f1_compressed
    os.system(cmd)
    # compress and resize video 2
    cmd = "ffmpeg -y -i " + f2 + " -s " + str(w) + "x" + str(h) + " -vcodec h264 -b:v 1000k -acodec mp2 " + f2_compressed
    os.system(cmd)
    # merge and remove sound
    cmd = "ffmpeg -y -i " + f1_compressed + " -i " + f2_compressed + " -an -r 60 -filter_complex hstack " + out
    os.system(cmd)