import streamlit as st
import requests

API_URL='http://127.0.0.1:8000/predict'

st.set_page_config(page_title="Titanic AI Predictor", page_icon="🚢", layout="centered")
st.title("🚢 Ứng dụng AI Dự đoán Sinh/Tử Tàu Titanic")


st.sidebar.header("📝 Nhập thông tin của bạn:")

raw_input = {
    'Pclass': st.sidebar.selectbox("Hạng vé (Pclass):", [1, 2, 3], format_func=lambda x: f"Hạng {x}"),
    'Sex': st.sidebar.radio("Giới tính:", ["Nam", "Nữ"]),
    'Age': st.sidebar.slider("Độ tuổi:", 1, 100, 25),
    'Fare': st.sidebar.number_input("Giá vé đã mua ($):", 0.0, 500.0, 32.0),
    'SibSp': st.sidebar.number_input("Số anh chị em/vợ chồng:", 0, 10, 0),
    'Parch': st.sidebar.number_input("Số cha mẹ/con cái:", 0, 10, 0),
    'Embarked': st.sidebar.selectbox("Cảng lên tàu:", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
}

if st.button("🔮 TIẾN HÀNH DỰ ĐOÁN PHÁN QUYẾT", use_container_width=True):
    with st.spinner('Cỗ máy đang phân tích dữ liệu...'):
        try:
            response=requests.post(API_URL,json=raw_input)
            response.raise_for_status()
            
            result=response.json()
            prediction_class=result['prediction_class']
            survival_prob=result['prediction_probability']
            st.markdown("---")
            if prediction_class == 1:
                st.success("🎉 **CHÚC MỪNG! BẠN ĐÃ SỐNG SÓT!**")
                st.info(f"📊 Tỷ lệ sinh tồn: **{survival_prob * 100:.2f}%**")
                st.balloons()
            else:
                st.error("💀 **RẤT TIẾC... BẠN ĐÃ CHÌM CÙNG CON TÀU!**")
                st.warning(f"📊 Cơ hội sống sót: **{survival_prob * 100:.2f}%**")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Lỗi kết nối: Không thể gọi đến AI Server. Vui lòng kiểm tra xem FastAPI (uvicorn) đã được bật chưa!")
        except Exception as e:
            st.error(f"🚨 Đã có lỗi xảy ra: {e}")