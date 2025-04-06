import tensorflow as tf
import numpy as np
from google.cloud import storage
from PIL import Image

Bucket_name="red-tomato-model-1"
class_names=['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

model = None

def down_blob(bucket_name,source_blob,destination_file):  # modeli yükleyecegiz blob(binary large object)
 storage_client = storage.Client() # Google Cloud Storage’a bağlanmak için bir istemci (client) oluşturur.
 bucket=storage_client.get_bucket(bucket_name) # GCS'deki  belirtilen isme sahip bucket’ı bulur. Bucket: GCS'deki klasör gibi bir alan.
 blob=bucket.blob(source_blob) #Bucket içindeki belirtilen dosya adını (source_blob_name) referans alarak bir "Blob" nesnesi oluşturur.
 blob.download_to_filename(destination_file) # blob'daki veriyi alır ve locale kaydeder.

 print(f"blob {source_blob} downloaded to {destination_file}")


def predict(request):
 global model
 if model is None:
  down_blob(
   Bucket_name,
   "models/ilovetomatoooo.h5",
   "/tmp/ilovetomatoooo.h5",
  )
  model = tf.keras.models.load_model("/tmp/ilovetomatoooo.h5")

  image=request.files['image']
  image=np.array(Image.open(image).convert("RGB").resize((256,256))) # np array donusturuldu. rgb 3 channel.
  image=image/255  # 0-255 -> 0-1

  img_array=tf.expand_dims(image,0)
  predictions=model.predict(img_array)
  predicted_class=class_names[np.argmax(predictions[0])]  # img_array sadece tek image oldugu için [0]#
  confidence=round(100*(np.max(predictions[0])),2)
  return {
    "class " : predicted_class,
    "confidence" : confidence
   }

