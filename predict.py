import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_lvl1 = load_model("model_level1.h5")
model_bio = load_model("model_bio.h5")
model_nonbio = load_model("model_nonbio.h5")

bio_classes = ["food_waste","leaf_waste","paper_waste","wood_waste"]
nonbio_classes = ["ewaste","metal_cans","plastic_bags","plastic_bottles"]

def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred1 = model_lvl1.predict(img_array)

    if np.argmax(pred1) == 0:
        pred2 = model_bio.predict(img_array)
        return "Biodegradable → " + bio_classes[np.argmax(pred2)]
    else:
        pred2 = model_nonbio.predict(img_array)
        return "Non-Biodegradable → " + nonbio_classes[np.argmax(pred2)]

print(predict("2_178.jpg"))