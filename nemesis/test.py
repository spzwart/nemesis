import os
fname = "nemesis_movie.mp4"
dir = "nemesis"
os.system(f"ffmpeg -r 170 -i snapshot_%d.png -c:v libx264 -preset slow -crf 18 -vf 'scale=1920:1080' -y {fname}")
