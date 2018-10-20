# install these dependencies
# sudo pip3 install opencv-contrib-python image_match elasticsearch scipy


import os
import re
import cv2


from image_match.goldberg import ImageSignature

def main():
    searchSimilarImage('/home/maitreya/Downloads/6332 Marcas/train','/home/maitreya/Downloads/6332 Marcas/test/2.bmp',10)



def searchSimilarImage(imgDirectoryPath, testImagePath, topSearch):
    # imgDirectoryPath : Directory path where all images stored
    # testImagePath : Image path of test sample
    # topSearch : Numeric value to show top # images

    distances = []
    path = imgDirectoryPath
    rx = re.compile(r'\.(bmp)')
    for path, dnames, fnames in os.walk(path):
        for x in fnames:
            if rx.search(x):
                imgpath = os.path.join(path, x)
                imgname = os.path.splitext(os.path.basename(imgpath))[0]
                gis = ImageSignature()
                a = gis.generate_signature(testImagePath)
                b = gis.generate_signature(imgpath)
                dis = (gis.normalized_distance(a, b))
                ar = {'name': imgname + '.bmp','path': imgpath, 'distance': dis}
                distances.append(ar)
    newlist = sorted(distances, key=lambda k: k['distance'])

    for i in range(topSearch):
        img = cv2.imread(newlist[i].get('path'))
        cv2.imshow(newlist[i].get('name'),img)
        print(newlist[i].get('path'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()