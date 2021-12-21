from flask import Flask, render_template, request , jsonify
import time
import os
from my_tools.index import index
from my_tools.index import index_one
from my_tools.search import Search


app = Flask(__name__)
#general parameters
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
params = {"theta": 4, "frequency": (0, 1, 0.5, 0.8), "sigma": (1, 3), "n_slice": 2}

#Database Offline Indexing
@app.route('/offlineIndex')
def test():
    index(params)
    return "Done !!"

#Index route
@app.route('/')
def index():
    return render_template('main.html' )

@app.post('/upload')
def upload():
    #Saving the Uploaded image in the Upload folder
    file = request.files['image']
    new_file_name = str(
        str(time.time()) + '.png'
    )
    file.save(os.path.join(
            app.config['UPLOAD_FOLDER'],new_file_name
        )
    )

    #Extracting the feature vetor from the uploaded images and adding this vector to our database
    features = index_one(str(UPLOAD_FOLDER + '/' + new_file_name) , params)

    #Comparing and sorting the uploaded image's features with the offline-calulcated images features
    searcher = Search('./index.csv')
    results = searcher.search(features)
    # results = searcher.gaborSearch(features)
    RESULTS_LIST = list()
    for (score, pathImage) in results:
        RESULTS_LIST.append(
            {"image": str(pathImage), "score": str(score)}
        )

    #returning the search results
    return jsonify(RESULTS_LIST)

if __name__ == '__main__':
    app.run(debug=True)