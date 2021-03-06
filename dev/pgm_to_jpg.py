from PIL import Image
import os, glob
 
def batch_image(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)
    
    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    count = 0
    for files in glob.glob(in_dir+'/*.pgm'):
        filepath, filename = os.path.split(files)
        print(in_dir)
        print(files)
        out_file = filename[0:19] + '.jpg'
        # print(filepath,',',filename, ',', out_file)
        im = Image.open(files)
        new_path = os.path.join(out_dir, out_file)
        print(count, ',', new_path)
        count = count + 1
        im.save(os.path.join(out_dir, out_file))
        
 
if __name__=='__main__':
    #create folder
    os.mkdir('./CroppedYale_jpg')
    #call function for every image folder
    for subdir, dirs, files in os.walk("./CroppedYale"):
        if subdir!='./CroppedYale':
    	    batch_image(subdir, './CroppedYale_jpg/'+subdir[14:])
