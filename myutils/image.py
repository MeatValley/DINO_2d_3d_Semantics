from PIL import Image
import PIL
import matplotlib.pyplot as plt

def load_image(path):
    """
    Read an image using PIL

    Parameters
    ----------
    path : str
        Path to the image

    Returns
    -------
    image : PIL.Image
        Loaded image
    """
    return Image.open(path)
    
if __name__ == "__main__":
    img = load_image('data\Area1\conferenceRoom_1\stream\depth\camera_0d600f92f8d14e288ddc590c32584a5a_conferenceRoom_1_frame_21_domain_depth.png')
    pixel_map = img.load()

    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):    # For every row
            if pixel_map[i,j]==0: print(pixel_map[i,j]) # set the colour accordingly

    img2 = Image.new( 'RGB', (1080,1080), "black") # create a new black image
    pixels = img.load() # create the pixel map


    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):
            # print(pixels[i,j])
            p = pixels[i,j]
            p0 = pixels[i,j-1]
            p1 = pixels[i,j+1]
            if (p> p0+15) or (p<p0) : pixels[i,j] = 0 
            
    img.show()
