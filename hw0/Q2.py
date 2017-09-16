from PIL import Image
import sys
__author__ = 'b04901025'


def main():
    filename = sys.argv[1]
    im = Image.open(filename)
    a = list(im.convert('RGB').getdata())
    b = []
    for pixel in a:
        pixel = list(pixel)
        pixel[0] = int(pixel[0]/2)
        pixel[1] = int(pixel[1]/2)
        pixel[2] = int(pixel[2]/2)
        pixel = tuple(pixel)
        b.append(pixel)

    im2 = Image.new(im.mode, im.size)
    im2.putdata(b)
    im2.save("Q2.jpg")

if __name__ == '__main__':
    main()
