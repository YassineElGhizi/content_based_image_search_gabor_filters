from flask import Flask, render_template, request , jsonify
import time
import os
import cv2
from my_tools.index import train
from my_tools.index import train_one
from my_tools.gabor import GaborDescriptor
from my_tools.search import Search


UPLOAD_FOLDER = 'static/images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
params = {"theta": 4, "frequency": (0, 1, 0.5, 0.8), "sigma": (1, 3), "n_slice": 2}


@app.route('/offlineIndex')
def test():
    train()
    return "Done !!"

@app.route('/')
def index():
    return render_template('main.html' )

@app.get('/test')
def test2():
    return "working !!"

@app.post('/upload')
def upload():
    file = request.files['image']
    new_file_name = str(
        str(time.time()) + '.png'
    )
    file.save(os.path.join(
            app.config['UPLOAD_FOLDER'],new_file_name
        )
    )

    train_one(str(UPLOAD_FOLDER + '/' + new_file_name))



    gd = GaborDescriptor(params)
    gaborKernels = gd.kernels()
    query = cv2.imread(os.path.join(os.path.dirname(__file__), 'static/images/' + new_file_name))
    features = gd.gaborHistogram(query, gaborKernels)

    #searching
    searcher = Search('./index.csv')
    # results = searcher.search(features)
    results = searcher.gaborSearch(features)
    RESULTS_LIST = list()
    for (score, pathImage) in results:
        RESULTS_LIST.append(
            {"image": str(pathImage), "score": str(score)}
        )

    return jsonify(RESULTS_LIST)


if __name__ == '__main__':
    app.run(debug=True)