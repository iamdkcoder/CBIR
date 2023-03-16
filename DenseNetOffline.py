from PIL import Image
from DensenetMod import FeatureExtractorD
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractorD()
    for img_path in sorted(Path("./static/image").glob("*.jpeg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/DenseNetFeatures") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
