import streamlit as st

def show_home2():
    st.title("แนวทางการพัฒนาโมเดลทำนายน้ำหนักจาก Obesity Prediction Dataset")
    st.write("""
    โครงการนี้พัฒนาขึ้นเพื่อสร้างแบบจำลองการเรียนรู้เชิงลึก (**Deep Learning Model**) ใช้ Machine Learning 
    สำหรับทำนายน้ำหนัก โดยใช้ **Linear Regression (LR)**
    """)

    st.header("📌 ขั้นตอนการพัฒนา")
    
    st.subheader("1️⃣ การเตรียมข้อมูล")
    st.write("""โมเดลนี้ใช้ข้อมูลสุขภาพจาก Kaggle ซึ่งมีข้อมูลของบุคคลเกี่ยวกับอายุ ส่วนสูง BMI และน้ำหนัก""")
    st.write("""
    - **Feature หลัก :** อายุ (Age), ส่วนสูง (Height), ดัชนีมวลกาย (BMI)
    - **ตัวแปรเป้าหมาย :** น้ำหนัก (Weight)
    """)

    st.subheader("2️⃣ ทฤษฎีของอัลกอริทึม Linear Regression")
    st.write("""**Linear Regression (LR)**""")
    st.write("""
    - ใช้สมการเชิงเส้น : 𝑊𝑒𝑖𝑔ℎ𝑡 = 𝑏0 + 𝑏1 ⋅ 𝐺𝑒𝑛𝑑𝑒𝑟 + 𝑏2 ⋅ 𝐻𝑒𝑖𝑔ℎ𝑡 + 𝑏3 ⋅ 𝑂𝑏𝑒𝑠𝑖𝑡𝑦
    - Loss Function: **Mean Squared Error (MSE)**
    - ใช้วิธี **Gradient Descent** ในการหาค่าพารามิเตอร์ที่ดีที่สุด
    - เหมาะกับการพยากรณ์ค่าตัวเลข เช่น น้ำหนักของบุคคล
    """)

    st.subheader("3️⃣ ขั้นตอนการพัฒนาโมเดล Linear Regression")
    st.subheader("การโหลดและเตรียมข้อมูล")
    st.code("""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        import joblib

        # โหลดและเตรียมข้อมูล
        weight_data = pd.read_csv('dataset/obesity_data.csv')
        X = weight_data[['Age', 'Height', 'BMI']]
        y = weight_data['Weight']

        # แบ่งข้อมูลเป็น training และ test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """)

    st.subheader("การสร้างและเทรนโมเดล Linear Regression")
    st.code("""
        # สร้างและเทรนโมเดล Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # ทดสอบและประเมินผล
        y_pred = lr_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'MSE: {mse}, R2 Score: {r2}')

        # บันทึกโมเดล
        joblib.dump(lr_model, 'linear_regression_model.pkl')
    """)

    st.subheader("4️⃣ การใช้งานโมเดล Linear Regression")
    st.code("""
        import joblib
        import numpy as np

        # โหลดโมเดล
        lr_model = joblib.load('linear_regression_model.pkl')

        # ทำนายผลใหม่
        new_data = np.array([[25, 170, 24.5]])  # อายุ 25, ส่วนสูง 170 cm, BMI 24.5
        predicted_weight = lr_model.predict(new_data)
        print(f'Predicted Weight: {predicted_weight[0]} kg')
    """)

    st.success("🎯 ระบบพร้อมใช้งานแล้ว! กรุณาเลือกเมนูจากแถบด้านซ้ายเพื่อทดลองใช้งาน")

    
