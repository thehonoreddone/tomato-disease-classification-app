from fastapi import FastAPI,File,UploadFile
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
from starlette.middleware.cors import CORSMiddleware

app=FastAPI()

origins = [
    "https://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model=tf.keras.models.load_model("../models/1.keras")
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

@app.get("/ping")
async def ping():
    return "hi guys bro"

def read_file_as_image(data) -> np.ndarray:
    img=np.array(Image.open(BytesIO(data)))
    return img
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image=read_file_as_image(await file.read())
    image=np.expand_dims(image,axis=0)

    prediction=model.predict(image)
    predicted_class=class_names[np.argmax(prediction[0])]
    confidence=float(np.max(prediction[0]))       # float olmalı fastpi json formatına dönüştürecegi icin hata alır.
    return{
        "class" : predicted_class,
        "confidence":confidence
    }









if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)