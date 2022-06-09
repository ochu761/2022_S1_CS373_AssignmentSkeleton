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
    bin_sum = [0 for i in range(nr_bins)]

    bin_width = math.ceil(255/nr_bins)
    
    for i in range(image_height):
        for j in range(image_width):
            # if max = 255, and nr_bins = 8
            # 255/8 = 31.875
            # if p = 255, 255/31.875 = 8
            
            bin_index = math.floor(pixel_array[i][j]/bin_width)
            
            histogram[bin_index] += 1
            bin_sum[bin_index] += pixel_array[i][j]

    return [histogram, bin_sum]

def convertToGreyscale(r, g, b, image_width, image_height):
    for y in range(image_height):
        for x in range(image_width):
            # reuse r as output array as will not be impacted by subsequent iterations
            r[y][x] = round(0.299*r[y][x] + 0.587*g[y][x] + 0.114*g[y][x])
    
    return r

def getLowestMeanChannel(channels):
    lowest_mean = 255
    lowest_channel = []

    for channel in channels:
        channel_mean = computeArrayMean(channel)
        if channel_mean < lowest_mean:
            lowest_mean = channel_mean
            lowest_channel = channel
    
    return lowest_channel

def getLargestSDChannel(channels):
    largest_SD = 0
    selected_channel = channels[0] # temp assignment

    for channel in channels:
        mean = computeArrayMean(channel)
        variances = []
        
        for row in range(len(channel)):
            for col in range(len(channel[0])):
                variances.append(math.pow(mean - channel[row][col], 2))
            
        channel_SD = math.sqrt(sum(variances)/len(variances))
        if channel_SD > largest_SD: 
            largest_SD = channel_SD
            selected_channel = channel
    
    return selected_channel

def computeArrayMean(array):
    return sum(sum(array, [])) / (len(array) * len(array[0]))


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



def simpleThresholdToBinary(px_array, image_width, image_height, threshold, min=0, max=255):
    for y in range(image_height):
        for x in range(image_width):
            # reuse px_array as output array as will not be impacted by subsequent iterations
            if px_array[y][x] <= threshold: px_array[y][x] = min
            else: px_array[y][x] = max
    
    return px_array

def adaptiveThresholdToBinary(px_array, image_width, image_height, min=0, max=255):
    nr_bins = 32
    [histogram, bin_sum] = computeHistogram(px_array, image_width, image_height, nr_bins)
    bin_width = math.ceil(255/nr_bins)

    threshold = sum(bin_sum) / sum(histogram)
    print(threshold)

    while True:
        threshold_index = math.floor(threshold/bin_width)

        lower_count = histogram[:threshold_index]
        lower_sum = bin_sum[:threshold_index]
        upper_count = histogram[threshold_index+1:]
        upper_sum = bin_sum[threshold_index+1:]

        lower_mean = sum(lower_sum) / sum(lower_count)
        upper_mean = sum(upper_sum) / sum(upper_count)

        new_threshold = 0.5 * (lower_mean + upper_mean)
        print(new_threshold)

        if new_threshold == threshold:
            threshold = new_threshold
            break

        threshold = new_threshold

    return simpleThresholdToBinary(px_array, image_width, image_height, threshold, min, max)




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

def computeLargestValidComponent(px_array, components, image_width, image_height):
    components = orderComponentsByLargest(components)
    dimension_shortlist = []
    alignment_shortlist = []

    # shortlist by dimensions
    for component in components:
        [min_x, max_x, min_y, max_y] = computeComponentBoundingBox(px_array, component[0], image_width, image_height)

        width = max_x - min_x
        height = max_y - min_y

        isValidRatio = isValidSize = False
        
        if height == 0: 
            ratio = 0
        else: 
            ratio = width / height

        if ratio >= 1.5 and ratio <= 5: isValidRatio = True
        if width > 0.1 * image_width or height > 0.1 * image_height: isValidSize = True

        if isValidRatio and isValidSize: dimension_shortlist.append(component)

    # sort by horizontal alignment
    for component in dimension_shortlist:
        component_val = component[0]
        [min_x, max_x, min_y, max_y] = computeComponentBoundingBox(px_array, component_val, image_width, image_height)
        longest_row_count = 0

        for y in range(min_y, max_y + 1):
            row_count = 0

            for x in range(min_x, max_x + 1):
                if px_array[y][x] == component_val: row_count += 1

            if row_count > longest_row_count: 
                longest_row_count = row_count

        # calculate horizontal alignment ratio and append to shortlist
        print(longest_row_count)
        print((component_val, longest_row_count/(max_x - min_x)))
        alignment_shortlist.append((component_val, longest_row_count/(max_x - min_x)))

    # sort by longest horizontal alignment ratio
    alignment_shortlist.sort(key=takeSecond, reverse=True)

    # return component value of component with greatest horizontal alignment ratio
    if (len(alignment_shortlist) == 0):
        return 0 # no component found

    print("Found largest valid component: " + str(alignment_shortlist[0][0]))
    return alignment_shortlist[0][0]

def orderComponentsByLargest(components):
    return sorted(components.items(), key=lambda item: item[1], reverse=True)

def takeSecond(element):
    return element[1]

   
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

    N_DILATIONS = 5
    N_EROSIONS = 5
    RECOMMENDED_THRESHOLD = 150

    input_filename = "numberplate3.png"

    # this is the default input image filename
    # D:E @ 150 - 5:5(123456)

    #TODO
    # - Consider trimming down identified component to edges by detecting min/max x and y with 89% straight edge boundary

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

    # STUDENT IMPLEMENTATION here

    # Convert to greyscale
    print("Converting image channels to single pixel array")
    #px_array = convertToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    #px_array = getLowestMeanChannel([px_array_r, px_array_g, px_array_b])
    px_array = getLargestSDChannel([px_array_r, px_array_g, px_array_b])

    initial_img = px_array

    # Contrast stretching
    print("Applying contrast stretching")
    px_array = contrastStretch(px_array, image_width, image_height)
   

    # High contrast filtering
    print("Computing standard deviation")
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    print("Applying contrast stretching")
    px_array = contrastStretch(px_array, image_width, image_height)

    # Thresholding for segmentation (to binary with threshold=150, min=0 and max=1)
    print("Computing and applying threshold")
    threshold_img = px_array = simpleThresholdToBinary(px_array, image_width, image_height, RECOMMENDED_THRESHOLD, 0, 1)
    #threshold_img = px_array = adaptiveThresholdToBinary(px_array, image_width, image_height)

    # Morphological operations
    print("Computing pre-dilation")
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    print("Computing pre-erosion")
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    for i in range(N_DILATIONS):
        print("Computing dilation " + str(i+1) + "/" + str(N_DILATIONS))
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    for i in range(N_EROSIONS):
        print("Computing erosion " + str(i+1) + "/" + str(N_EROSIONS))
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    morph_img = px_array

    # problem: images where the licence plate is small are more susceptible to "strings"
    bbox_min_x = bbox_max_x = bbox_min_y = bbox_max_y = 0
    prev_area = -1
    area = 0

    while prev_area != area:
        # Connected component analysis
        print("Computing connected components")
        [px_array, components] = computeConnectedComponentLabeling(px_array, image_width, image_height)

        component_img = px_array
        component = computeLargestValidComponent(px_array, components, image_width, image_height)

        # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
        print("Computing component bounding box")
        [bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y] = computeComponentBoundingBox(px_array, component, image_width, image_height)

        prev_area = area

        # if component is small, consider opening to reduce undue influence by small SE

        if bbox_max_x - bbox_min_x < 0.4 * image_width or bbox_max_x - bbox_min_x < 0.4 * image_height:    
            print("Component is small: computing opening (erosion followed by dilation)")
            px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
            px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

            area = (bbox_max_x - bbox_min_x) * (bbox_max_y - bbox_min_y)

    # Draw a bounding box as a rectangle into the input image
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='r', facecolor='none')

    # setup the plots for intermediate results in a figure
    DISPLAY_MODE = 1

    if DISPLAY_MODE == 0:
        fig1, axs1 = pyplot.subplots(2, 2)
        axs1[0, 0].set_title('Input red channel of image')
        axs1[0, 0].imshow(px_array_r, cmap='gray')
        axs1[0, 1].set_title('Input green channel of image')
        axs1[0, 1].imshow(px_array_g, cmap='gray')
        axs1[1, 0].set_title('Input blue channel of image')
        axs1[1, 0].imshow(px_array_b, cmap='gray')

        axs1[1, 1].set_title('Final image of detection')
        axs1[1, 1].imshow(px_array, cmap='gray')

        axs1[1, 1].add_patch(rect)
    elif DISPLAY_MODE == 1:
        fig1, axs1 = pyplot.subplots(2, 2)
        axs1[0,0].set_title('Threshold')
        axs1[0,0].imshow(threshold_img, cmap='gray')
        axs1[0,1].set_title('Morphological operations')
        axs1[0,1].imshow(morph_img, cmap='gray')
        axs1[1,0].set_title('Component labels')
        axs1[1,0].imshow(component_img, cmap='gray')
        axs1[1,1].set_title('Final image of detection')
        axs1[1,1].imshow(initial_img, cmap='gray')

        axs1[1,1].add_patch(rect)
        

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1,1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()