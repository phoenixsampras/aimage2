# install these dependencies
# sudo pip3 install opencv-contrib-python image_match elasticsearch scipy


import os
import re
import cv2
import numpy as np
import pickle
from pathlib import Path
from image_match.goldberg import ImageSignature

# Flask stuff
from PIL import Image
from flask import Flask, request, render_template
from datetime import datetime

app = Flask(__name__)
FLASK_DEBUG = 1

def createDB(imgDirectoryPath,dbFilePath):
    distances = []
    path = imgDirectoryPath
    rx = re.compile(r'\.(bmp)')
    for path, dnames, fnames in os.walk(path):
        for x in fnames:
            if rx.search(x):
                imgpath = os.path.join(path, x)
                print('Creating index...',imgpath)
                imgname = os.path.splitext(os.path.basename(imgpath))[0]
                img = cv2.imread(imgpath)
                dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
                median = cv2.medianBlur(dst, 7)
                gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
                th, im_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((7, 7), dtype=np.uint8)
                binary_cleaned = cv2.morphologyEx(im_thresh, cv2.MORPH_OPEN, kernel)
                localimg = np.array(binary_cleaned)
                ar = {'name': imgname + '.bmp', 'path': imgpath, 'data': localimg}
                distances.append(ar)

    with open(dbFilePath, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(distances, filehandle)

def doSearch(dbFilePath,imgDirectoryPath, testImagePath,topSearch ):

    # dbFilePath : Database data file path
    # imgDirectoryPath : Directory path where all images stored
    # testImagePath : Image path of test sample
    # topSearch : Numeric value to show top # images
    fixeddistances = []
    db_file = Path(dbFilePath)
    if db_file.is_file():
        print('Found DB File')
        fixeddistances = searchSimilarImages(dbFilePath, testImagePath, topSearch)
    else:
        print('Not Found DB File... Creating new One')
        createDB(imgDirectoryPath, dbFilePath)
        fixeddistances = searchSimilarImages(dbFilePath,testImagePath,topSearch)

    return fixeddistances

def searchSimilarImages(dbFilePath, testImagePath,topSearch):
    print('Starting Search')
    distancemap = []
    img = cv2.imread(testImagePath)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    median = cv2.medianBlur(dst, 7)
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    th, im_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), dtype=np.uint8)
    binary_cleaned = cv2.morphologyEx(im_thresh, cv2.MORPH_OPEN, kernel)
    testdata = np.array(binary_cleaned)

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


# if __name__ == '__main__':
#     main()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/upload/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)
        scores = doSearch('static/localdb.data','static/6332', uploaded_img_path, 10)
        print(scores)
        return render_template('index.html',query_path=uploaded_img_path,scores=scores)
    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run("0.0.0.0")
