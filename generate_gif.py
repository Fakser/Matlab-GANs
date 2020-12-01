import imageio
import glob
from sys import argv
from tqdm import tqdm

anim_file = 'gif.gif'
pic_dir = './iterations_images/anime_faces_*.png'
if len(argv) > 1: 
    for i in range(len(argv)):        
        if argv[i] == '-o' and i < len(argv) - 1:
            anim_file = argv[i+1]
        if argv[i] == '-dir' and i < len(argv) - 1:
            pic_dir = argv[i+1]

            
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(pic_dir)
    filenames = sorted(filenames, key = lambda x: int(x.replace('./iterations_images\\anime_faces_', '').replace('.png', '')))
    last = -1
    for i,filename in tqdm(enumerate(filenames)):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)