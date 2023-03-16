import numpy as np
from PIL import Image
from DensenetMod import FeatureExtractorD
from EfficientNetMod import FeatureExtractorE
from MobileNetMod import FeatureExtractorM
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from io import FileIO


app = Flask(__name__)

# Read image features
DenseFeature = FeatureExtractorD()
EfficientFeature = FeatureExtractorE()
MobileFeature = FeatureExtractorM()
DenseDF = []
EffiDF=[]
DenseEffDF=[]
MobiDF=[]
img_paths = []
for feature_path in Path("./static/DenseNetFeatures").glob("*.npy"):
    DenseDF.append(np.load(feature_path))
    img_paths.append(Path("./static/image") / (feature_path.stem + ".jpeg"))
DenseDF = np.array(DenseDF)

for feature_path in Path("./static/EfficientNetFeatures").glob("*.npy"):
    EffiDF.append(np.load(feature_path))

EffiDF = np.array(EffiDF)

for feature_path in Path("./static/DenseEffNetFeatures").glob("*.npy"):
    DenseEffDF.append(np.load(feature_path))
DenseEffDF = np.array(DenseEffDF)

for feature_path in Path("./static/MobileNetFeatures").glob("*.npy"):
    MobiDF.append(np.load(feature_path))
MobiDF = np.array(MobiDF)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        scope = request.form.get('scope')
        FeatureModel = request.form.get('feature_select')
        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        if FeatureModel=='DenseNet':
            query = DenseFeature.extract(img)
            dists = np.linalg.norm(DenseDF-query, axis=1)  # L2 distances to features
        elif FeatureModel=='EfficientNet':
            query=EfficientFeature.extract(img)
            dists = np.linalg.norm(EffiDF-query, axis=1)  # L2 distances to features
        elif FeatureModel=='MobileNet':
            query=MobileFeature.extract(img)
            dists = np.linalg.norm(MobiDF-query, axis=1)  # L2 distances to features
        elif FeatureModel=='DenseEffNet':
            queryA = DenseFeature.extract(img)
            queryB=EfficientFeature.extract(img)
            query=np.concatenate((queryA,queryB))
            dists = np.linalg.norm(np.subtract(DenseEffDF,query), axis=1)  # L2 distances to features

        # Run search
        scope=int(scope)
        ids = np.argsort(dists)[1:scope+1]  # Top 10 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        name_path=[]
        for id in scores:
            nam=id[1]
            nam=repr(nam)
            idx=nam.index("_")
            nam=nam[26:idx]
            name_path.append(nam)
        inx=uploaded_img_path.index("_")
        inp=uploaded_img_path[inx+1:]
        inx=inp.index("_")
        inp=inp[0:inx]
        c=name_path.count(inp) 
        recall = c/200
        precision=c/scope
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores,precision=precision,recall=recall)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
