import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# ======== STUDENT IMPLEMENTATION

def computeHistogram(pixel_array, image_width, image_height, nr_bins):

    histogram = [0 for i in range(nr_bins)]

    binWidth = math.ceil(255/nr_bins)
    
    for i in range(image_height):
        for j in range(image_width):
            # if max = 255, and nr_bins = 8
            # 255/8 = 31.875
            # if p = 255, 255/31.875 = 8
            
            histogram[math.floor(pixel_array[i][j]/binWidth)] += 1

    return histogram

def convertToGreyscale(r, g, b, image_width, image_height):
    for y in range(image_height):
        for x in range(image_width):
            # reuse r as output array as will not be impacted by subsequent iterations
            r[y][x] = round(0.299*r[y][x] + 0.587*g[y][x] + 0.114*g[y][x])
    
    return r


def contrastStretch(px_array, image_width, image_height):

    [min, max] = computeMinAndMaxValues(px_array, image_width, image_height)
    
    # case where contrast stretching is not possible
    if (min == max): return px_array
    
    # otherwise, contrast stretch and scale to 255
    for y in range(image_height):
        for x in range(image_width):
            # reuse px_array as output array as will not be impacted by subsequent iterations
            sout = round((px_array[y][x] - min)*(255/(max-min)))
            
            if sout < 0: px_array[y][x] = 0
            elif sout > 255: px_array[y][x] = 255
            else: px_array[y][x] = sout
    
    return px_array

def computeMinAndMaxValues(px_array, image_width, image_height):
    min = 255
    max = 0
    
    for y in range(image_height):
        for x in range(image_width):
            curVal = px_array[y][x]
            if curVal < min:
                min = curVal
            if curVal > max:
                max = curVal

    return [min, max]



def computeStandardDeviationImage5x5(px_array, image_width, image_height):
    # using BorderIgnore for a 2px boundary
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for y in range(2, image_height - 2):
        for x in range(2, image_width - 2):
            neighbours = []
    
            # get neighbour pixels of current pixel x,y
            for row in range(-2,3):
                for col in range(-2,3):
                    neighbours.append(px_array[y+row][x+col])
            
            # compute standard deviation of neighbourhood
            mean = sum(neighbours)/len(neighbours)
            variances = []
            
            for nb in neighbours:
                variances.append(math.pow(mean - nb, 2))
                
            new_array[y][x] = math.sqrt(sum(variances)/len(neighbours))
            
    return new_array



def simpleThresholdToBinary(px_array, image_width, image_height, threshold=150, min=0, max=255):
    for y in range(image_height):
        for x in range(image_width):
            # reuse px_array as output array as will not be impacted by subsequent iterations
            if px_array[y][x] <= threshold: 
                px_array[y][x] = min
            else: px_array[y][x] = max
    
    return px_array

def adaptiveThresholdToBinary(px_array, image_width, image_height, threshold=150):
    #TODO
    pass




def computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height):
    # assume is binary image with min=0, max=1
    # using BorderIgnore for a 1px boundary
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            
            hits = False
            
            # for each neighbour
            for row in range(-1,2):
                for col in range(-1,2):
                    if px_array[y+row][x+col] != 0: hits = True
            
            if hits: new_array[y][x] = 1
    
    return new_array  

def computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height):
    # assume is binary image with min=0, max=1
    # using BorderIgnore for a 1px boundary
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            
            fits = True
            
            # for each neighbour
            for row in range(-1,2):
                for col in range(-1,2):
                    if px_array[y+row][x+col] == 0: fits = False
            
            if fits: new_array[y][x] = 1
    
    return new_array



def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    curLabel = 1
    components = {}
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    pixel_obj = [ [0]*image_width for i in range(image_height)]
    
    # create Pixel instance for each pixel
    for y in range(image_height):
        for x in range(image_width):
            pixel_obj[y][x] = Pixel(x,y,pixel_array[y][x])
    
    for y in range(image_height):
        for x in range(image_width):
            if pixel_obj[y][x].isObj and not pixel_obj[y][x].isVisited:
                q = Queue()

                pixel_obj[y][x].isVisited = True
                components.update({curLabel: 0})

                q.enqueue(pixel_obj[y][x])
                
                while not q.isEmpty():
                    front_pixel = q.dequeue()

                    front_pixel.label = curLabel
                    components.update({curLabel: components[curLabel] + 1})

                    new_array[front_pixel.y][front_pixel.x] = curLabel
                    
                    f_x = front_pixel.x
                    f_y = front_pixel.y

                    # check neighbours
                    neighbour_index = [ [-1,0],[1,0],[0,1],[0,-1] ]

                    for iPos in neighbour_index:
                        if isInBounds(f_x+iPos[0],f_y+iPos[1],image_width,image_height):
                            p = pixel_obj[f_y+iPos[1]][f_x+iPos[0]]

                            if p.isObj and not p.isVisited: 
                                p.isVisited = True
                                q.enqueue(p)
                
                curLabel += 1
                
    return [new_array, components]

def isInBounds(x,y,width,height):
    if x >= 0 and x < width and y >= 0 and y < height:
        return True
    
    return False

def isolateLargestComponent(px_array, components, image_width, image_height, min=0, max=255):
    largest_component = computeLargestComponent(components)
    
    for y in range(image_height):
        for x in range(image_width):
            if px_array[y][x] != largest_component: px_array[y][x] = min
            else: px_array[y][x] = max

    return px_array

def computeComponentBoundingBox(px_array, component, image_width, image_height):
    min_x = image_width
    max_x = 0
    min_y = image_height
    max_y = 0

    for y in range(image_height):
        for x in range(image_width):
            if px_array[y][x] == component:
                if x > max_x: max_x = x
                if x < min_x: min_x = x
                
                if y > max_y: max_y = y
                if y < min_y: min_y = y

    return [min_x, max_x, min_y, max_y]

def computeLargestComponent(components):
    largest_count = 0
    largest_component = 0 # null value, labelling always starts at 1 (see computeConnectedComponentLabeling)

    for component, count in components.items():
        if count > largest_count:
            largest_count = count
            largest_component = component

    return largest_component




   
class Pixel:
    def __init__(self, x, y, val):
        self.x = x
        self.y = y
        self.val = val
        self.label = 0
        
        if (val > 0): self.isObj = True
        else: self.isObj = False
        
        self.isVisited = False


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)





# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate1.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    # Convert to greyscale
    px_array = convertToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    greyscale_img = px_array

     # Contrast stretching
    px_array = contrastStretch(px_array, image_width, image_height)
   

    # High contrast filtering
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    px_array = contrastStretch(px_array, image_width, image_height)

    # Thresholding for segmentation (to binary with threshold=150, min=0 and max=1)
    px_array = simpleThresholdToBinary(px_array, image_width, image_height, 150, 0, 1)

    # Morphological operations (repeat N_MORPH_OPS times for each)
    N_MORPH_OPS = 3

    for i in range(N_MORPH_OPS):
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    for i in range(N_MORPH_OPS):
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    # Connected component analysis
    [px_array, components] = computeConnectedComponentLabeling(px_array, image_width, image_height)
    #px_array = isolateLargestComponent(px_array, components, image_width, image_height)
    

    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # center_x = image_width / 2.0
    # center_y = image_height / 2.0
    # bbox_min_x = center_x - image_width / 4.0
    # bbox_max_x = center_x + image_width / 4.0
    # bbox_min_y = center_y - image_height / 4.0
    # bbox_max_y = center_y + image_height / 4.0
    [bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y] = computeComponentBoundingBox(px_array, computeLargestComponent(components), image_width, image_height)





    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(greyscale_img, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()