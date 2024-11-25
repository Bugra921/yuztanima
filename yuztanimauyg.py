import streamlit as st
import cv2
from PIL import Image
import io
from simple_facerec import SimpleFacerec  # Yüz tanıma kütüphanesi
import numpy as np

# SimpleFacerec sınıfını oluşturup yüzleri yükleme
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Yüz verilerinin olduğu klasör

# Streamlit arayüzü
st.title('Streamlit Yüz Tanıma Uygulaması')

# Kamera inputu ile resim çekme
cekilenresim = st.camera_input('Kameradan Resim Çek')

if cekilenresim is not None:
    # Resmi PIL formatından OpenCV formatına dönüştürme
    img_bytes = cekilenresim.getvalue()
    img = Image.open(io.BytesIO(img_bytes))
    img_cv2 = np.array(img)  # PIL'den OpenCV formatına çevirme (NumPy dizisi)

    # OpenCV ile yüz tanıma işlemi
    face_locations, face_names = sfr.detect_known_faces(img_cv2)
    
    # Tanınan yüzlerin etrafına dikdörtgen çizme ve isim yazma
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(img_cv2, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # OpenCV formatındaki görüntüyü tekrar Streamlit'e uygun hale getirme
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption='Tanınan Yüzler', use_column_width=True)

