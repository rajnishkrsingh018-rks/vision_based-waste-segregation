import tensorflow as tf

# Fix level1 model
model = tf.keras.models.load_model("model_level1.h5", compile=False)
model.save("model_level1_fixed.h5")

# Fix bio model
model = tf.keras.models.load_model("model_bio.h5", compile=False)
model.save("model_bio_fixed.h5")

# Fix nonbio model
model = tf.keras.models.load_model("model_nonbio.h5", compile=False)
model.save("model_nonbio_fixed.h5")

print("Models fixed and saved!")