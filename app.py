import streamlit as st

# ตั้งค่าหน้าเว็บ (ต้องเป็นคำสั่งแรกสุดของสคริปต์)
st.set_page_config(layout="wide", page_title="Cat vs Dog Classifier", page_icon="🐶")

import home  # Import หน้าแรก
import home2 
import app2 as app2  # Import หน้า app2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# โหลดโมเดล TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# หาค่า index ของ input และ output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(img_path):
    """ฟังก์ชันสำหรับทำนายภาพที่อัปโหลด"""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # ใส่ค่าเข้าไปใน Tensor ของ TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # ดึงค่าผลลัพธ์ออกมา
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return "🐶 สุนัข" if prediction[0][0] > 0.5 else "🐱 แมว"

    
# สร้าง Sidebar Menu (กำหนดค่าเริ่มต้นที่หน้าแรก)
with st.sidebar:
    st.title("📌 เมนู")
    page = st.selectbox("เลือกเมนู", ["Neural Network Model", "Machine Learning Model", "🔍 ทำนายภาพ", "⚖️ ทำนายน้ำหนัก"], index=0) # index=0 ทำให้เริ่มที่หน้าแรก

# แสดงเนื้อหาตามเมนูที่เลือก
if page == "Neural Network Model":
    home.show_home()

elif page == "Machine Learning Model":
    home2.show_home2()

elif page == "🔍 ทำนายภาพ":
    st.title("🐶🐱 Cat vs Dog Classifier")
    st.write("อัปโหลดภาพแมวหรือสุนัข ระบบจะทำการจำแนกประเภทให้")
    
    uploaded_file = st.file_uploader("📂 เลือกไฟล์ภาพ", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="📷 ภาพที่อัปโหลด", use_column_width=True)
        st.write("🔍 กำลังประมวลผล...")
        
        # บันทึกไฟล์ชั่วคราวเพื่อทำนาย
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        result = predict_image("temp.jpg")
        st.subheader(f"🎯 ผลลัพธ์: {result}")

elif page == "⚖️ ทำนายน้ำหนัก":
    app2.show_app2()


