import glob
from PIL import Image
import os
import natsort
def make_gif(frame_folder):
    frames = [Image.open(image) for image in natsort.natsorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save('2019_Carbon_Dioxide.gif', format="GIF", append_images=frames,
               save_all=True, duration=50, loop=0)

def get_frames(start_date, end_date,)
if __name__ == "__main__":
    make_gif(os.path.join(os.getcwd(),'2019'))