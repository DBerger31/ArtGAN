import cv2
import matplotlib.pyplot as plt

# creates an super res object
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "./web/ArtGAN/FSRCNN_x4.pb"

# read and creates the model
sr.readModel(path)
sr.setModel("fsrcnn", 4)

imgbase = 'rawimg/img'
suffix = '.png'

for i in range(1, 101):
    if str(i).endswith("1"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/abstract/res{i}.png", result)
    elif str(i).endswith("2"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/cityscape/res{i}.png", result)
    elif str(i).endswith("3"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/genrepainting/res{i}.png", result)
    elif str(i).endswith("4"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/illustration/res{i}.png", result)
    elif str(i).endswith("5"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/landscape/res{i}.png", result)
    elif str(i).endswith("6"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/portrait/res{i}.png", result)
    elif str(i).endswith("7"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/religiouspainting/res{i}.png", result)
    elif str(i).endswith("8"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/sketchandstudy/res{i}.png", result)
    elif str(i).endswith("9"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/stilllife/res{i}.png", result)
    elif str(i).endswith("0"):
        img = cv2.imread(f"{imgbase}{i}{suffix}")
        result = sr.upsample(img)
        cv2.imwrite(f"upscaled/symbolicpainting/res{i}.png", result)
