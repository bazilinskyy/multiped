import os
for x in range(4, 63):
    f1 = '../public/videos/video_' + str(x) + '.mp4'
    f2 = '../public/videos/video3rd_' + str(x) + '.mp4' 
    out = '../public/videos/videomerged_' + str(x) + '.mp4'
    os.system("ffmpeg -i " + f1 + " -i " + f2 + " -r 30 -filter_complex \"[0:v]scale=640:480, setpts=PTS-STARTPTS, pad=1280:720:0:120[left]; [1:v]scale=640:480, setpts=PTS-STARTPTS, pad=640:720:0:120[right]; [left][right]overlay=w; amerge,pan=stereo:c0<c0+c2:c1<c1+c3\" -vcodec libx264 -acodec aac -strict experimental " + out)