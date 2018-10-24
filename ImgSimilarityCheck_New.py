# install these dependencies
# sudo pip3 install opencv-contrib-python image_match elasticsearch scipy skimage


import os
import re
import cv2
import numpy as np
import pickle
from pathlib import Path
from skimage import morphology
from image_match.goldberg import ImageSignature
from scipy.spatial.distance import directed_hausdorff
import scipy.misc

# Flask stuff
from PIL import Image
from flask import Flask, request, render_template
from datetime import datetime

app = Flask(__name__)
FLASK_DEBUG = 1


def createDB(imgDirectoryPath, dbFilePath):
    distances = []
    path = imgDirectoryPath
    rx = re.compile(r'\.(bmp)')
    for path, dnames, fnames in os.walk(path):
        for x in fnames:
            if rx.search(x):
                imgpath = os.path.join(path, x)
                print('Creating index...', imgpath)
                imgname = os.path.splitext(os.path.basename(imgpath))[0]
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
                median = cv2.medianBlur(dst, 3)
                gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
                dim = (gray.shape)
                minsize = 0
                if dim[0] > 200:
                    minsize = 200
                else:
                    minsize = 100
                th, img_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
                binary_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
                binary_mask = cv2.bitwise_not(binary_cleaned)
                imglab = morphology.label(binary_mask)
                cleaned = morphology.remove_small_objects(imglab, min_size=minsize, connectivity=8)
                img3 = np.zeros((cleaned.shape))
                img3[cleaned > 0] = 255
                img3 = np.uint8(img3)
                clean_image = cv2.bitwise_not(img3)
                localimg = np.array(clean_image)
                # ar = {'name': imgname + '.bmp', 'path': 'static/6332/'+imgname + '.bmp', 'data': localimg}
                ar = {'name': imgname + '.bmp', 'path': imgpath, 'data': localimg}
                distances.append(ar)

    with open(dbFilePath, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(distances, filehandle)


def doSearch(dbFilePath, imgDirectoryPath, testImagePath, searchType='2', topSearch=10):
    # dbFilePath : Database data file path
    # imgDirectoryPath : Directory path where all images stored
    # testImagePath : Image path of test sample
    # topSearch : Numeric value to show top # images
    fixeddistances = []
    if searchType == '1':
        print('Running SCIPY Measurement')
        db_file = Path(dbFilePath)
        if db_file.is_file():
            print('Found DB File')
            fixeddistances = searchSimilarImages(dbFilePath, testImagePath, topSearch)
        else:
            print('Not Found DB File... Creating new One')
            createDB(imgDirectoryPath, dbFilePath)
            fixeddistances = searchSimilarImages(dbFilePath, testImagePath, topSearch)
    else:
        print('Running Hausdorff Measurement')
        db_file = Path(dbFilePath)
        if db_file.is_file():
            print('Found DB File')
            fixeddistances = searchSimilarImagesHausdorff(dbFilePath, testImagePath, topSearch)
        else:
            print('Not Found DB File... Creating new One')
            createDB(imgDirectoryPath, dbFilePath)
            fixeddistances = searchSimilarImagesHausdorff(dbFilePath, testImagePath, topSearch)

    return fixeddistances


def searchSimilarImages(dbFilePath, testImagePath, topSearch):
    print('Starting Search')
    distancemap = []
    img = cv2.imread(testImagePath, cv2.IMREAD_COLOR)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    median = cv2.medianBlur(dst, 3)
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    dim = (gray.shape)
    minsize = 0
    if dim[0] > 200:
        minsize = 200
    else:
        minsize = 100
    th, img_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    binary_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    binary_mask = cv2.bitwise_not(binary_cleaned)
    imglab = morphology.label(binary_mask)
    cleaned = morphology.remove_small_objects(imglab, min_size=minsize, connectivity=8)
    img3 = np.zeros((cleaned.shape))
    img3[cleaned > 0] = 255
    img3 = np.uint8(img3)
    clean_image = cv2.bitwise_not(img3)
    testdata = np.array(clean_image)

    with open(dbFilePath, 'rb') as filehandle:
        # read the data as binary data stream
        distances = pickle.load(filehandle)
        for i in range(len(distances)):
            imgname = distances[i].get('name')
            imgpath = distances[i].get('path')
            localdata = distances[i].get('data')
            gis = ImageSignature()
            a = gis.generate_signature(testdata)
            b = gis.generate_signature(localdata)
            dis = (gis.normalized_distance(a, b))
            ar = {'name': imgname + '.bmp', 'path': imgpath, 'distance': dis}
            distancemap.append(ar)

    newlist = sorted(distancemap, key=lambda k: k['distance'])
    fixeddistances = []
    for i in range(topSearch):
        fixeddistances.append(newlist[i])
    return fixeddistances


def searchSimilarImagesHausdorff(dbFilePath, testImagePath, topSearch):
    print('Starting Search')
    distancemap = []
    img = cv2.imread(testImagePath, cv2.IMREAD_COLOR)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    median = cv2.medianBlur(dst, 3)
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    dim = (gray.shape)
    minsize = 0
    if dim[0] > 200:
        minsize = 200
    else:
        minsize = 100
    th, img_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    binary_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    binary_mask = cv2.bitwise_not(binary_cleaned)
    imglab = morphology.label(binary_mask)
    cleaned = morphology.remove_small_objects(imglab, min_size=minsize, connectivity=8)
    img3 = np.zeros((cleaned.shape))
    img3[cleaned > 0] = 255
    img3 = np.uint8(img3)
    clean_image = cv2.bitwise_not(img3)
    testdata = np.array(clean_image).astype(float)

    with open(dbFilePath, 'rb') as filehandle:
        # read the data as binary data stream
        distances = pickle.load(filehandle)
        for i in range(len(distances)):
            imgname = distances[i].get('name')
            imgpath = distances[i].get('path')
            localdata = distances[i].get('data').astype(float)
            #localdata = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            dis = max(directed_hausdorff(testdata, localdata)[0],directed_hausdorff(localdata,testdata)[0])
            ar = {'name': imgname + '.bmp', 'path': imgpath, 'distance': dis}
            distancemap.append(ar)

    newlist = sorted(distancemap, key=lambda k: k['distance'])
    fixeddistances = []
    for i in range(topSearch):
        fixeddistances.append(newlist[i])
    return fixeddistances


# if __name__ == '__main__':
#     main()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        search = request.form['search_type']       
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/upload/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)
        scores = doSearch('static/localdb.data', 'static/6332', uploaded_img_path, search, 10)
        print(scores)
        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0")
