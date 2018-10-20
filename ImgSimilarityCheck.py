# install these dependencies
# sudo pip3 install opencv-contrib-python image_match elasticsearch scipy


import os
import re
import cv2

# Flask stuff
from PIL import Image
from flask import Flask, request, render_template
from datetime import datetime
app = Flask(__name__)
FLASK_DEBUG=1

from image_match.goldberg import ImageSignature

# def main():
#     searchSimilarImage('/mnt/c/Users/Phoenix Sampras/Documents/bb/fegasacruz/6332','/mnt/c/Users/Phoenix Sampras/Documents/bb/fegasacruz/test/2.bmp',10)

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
                print(imgname)
                gis = ImageSignature()
                a = gis.generate_signature(testImagePath)
                b = gis.generate_signature(imgpath)
                dis = (gis.normalized_distance(a, b))
                ar = {'name': imgname + '.bmp','path': imgpath, 'distance': dis}
                distances.append(ar)
    return sorted(distances, key=lambda k: k['distance'])
    # return [(dists[id], img_paths[id]) for id in ids]


    # for i in range(topSearch):
    #     img = cv2.imread(newlist[i].get('path'))
    #     cv2.imshow(newlist[i].get('name'),img)
    #     print(newlist[i].get('path'))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/upload/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)

        scores = searchSimilarImage('static/6332',uploaded_img_path,10)


        print (scores)


        # query = fe.extract(img)
        # dists = np.linalg.norm(features - query, axis=1)  # Do search
        # ids = np.argsort(dists)[:300] # Top 300 results
        # scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")
