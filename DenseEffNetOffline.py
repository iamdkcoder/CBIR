from PIL import Image
from DensenetMod import FeatureExtractorD
from EfficientNetMod import FeatureExtractorE
from pathlib import Path
import numpy as np
import time

if __name__ == '__main__':
    DenseFeatures = FeatureExtractorD()
    EfficientFeatures=FeatureExtractorE()
    start = time.time()
    for img_path in sorted(Path("./static/image").glob("*.jpeg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature1 = DenseFeatures.extract(img=Image.open(img_path))
        feature2 = EfficientFeatures.extract(img=Image.open(img_path))
        feature=np.concatenate((feature1,feature2))
        feature_path = Path("./static/DenseEffNetFeatures") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

    end = time.time()
    print((end-start)/60)