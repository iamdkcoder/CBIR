from PIL import Image
from MobileNetMod import FeatureExtractorM
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    fe = FeatureExtractorM()
    for img_path in sorted(Path("./static/image").glob("*.jpeg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/MobileNetFeatures") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)