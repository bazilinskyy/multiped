import os
for x in range(0, 63):
    f1 = 'video_' + str(x) + '.mp4'
    f2 = 'video3rd_' + str(x) + '.mp4' 
    out = 'videomerged_' + str(x) + '.mp4'
    w = 3840
    h = 2160
    cmd = "ffmpeg -i " + f1 + " -i " + f2 + " -r 60 -filter_complex hstack " + out
    os.system(cmd)

for x in range(64, 123):
    f1 = 'video_' + str(x) + '.mp4'
    f2 = 'video3rd_' + str(x-64) + '.mp4' 
    out = 'videomerged_' + str(x) + '.mp4'
    w = 3840
    h = 2160
    cmd = "ffmpeg -i " + f1 + " -i " + f2 + " -r 60 -filter_complex hstack " + out
    os.system(cmd)