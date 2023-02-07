import subprocess
import os

subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', r"rgb_%d.png",  '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '../output.mp4'], cwd=os.path.realpath('./video'))