from PIL import Image
from EfficientNetMod import FeatureExtractorE
from pathlib import Path
import numpy as np
import time

if __name__ == '__main__':
    fe = FeatureExtractorE()
    for img_path in sorted(Path("./static/image").glob("*.jpeg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/EfficientNetFeatures") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)