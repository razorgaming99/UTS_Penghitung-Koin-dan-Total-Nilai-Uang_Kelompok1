import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Penghitung Koin & Total Uang",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 Aplikasi Penghitung Koin dan Total Nilai Uang")
st.write("Unggah gambar koin untuk menghitung jumlah koin dan mendeteksi total nilai uang.")

# Sidebar untuk Pengaturan Parameter
st.sidebar.header("Pengaturan Deteksi (OpenCV)")
min_dist = st.sidebar.slider("Min Distance (HoughCircles)", 10, 150, 50)
param1 = st.sidebar.slider("Param 1 (Canny Threshold)", 10, 300, 100)
param2 = st.sidebar.slider("Param 2 (Accumulator Threshold)", 10, 150, 30)
min_radius = st.sidebar.slider("Min Radius", 0, 200, 20)
max_radius = st.sidebar.slider("Max Radius", 0, 300, 120)

def process_and_count_coins(image_np, min_dist, param1, param2, min_radius, max_radius):
    # Konversi ke Grayscale & Blur
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Deteksi Lingkaran Menggunakan HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    output_img = image_np.copy()
    total_coins = 0
    total_value = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        total_coins = len(circles[0, :])

        for i, c in enumerate(circles[0, :]):
            x, y, r = c[0], c[1], c[2]
            
            # Gambar lingkaran luar dan titik pusat koin
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 3)
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)
            
            # Pengkategorian koin berdasarkan ukuran radius (dapat disesuaikan)
            # Contoh logika penentuan nilai koin berdasarkan ukuran radius:
            if r > 70:
                value = 1000
            elif r > 50:
                value = 500
            elif r > 35:
                value = 200
            else:
                value = 100
                
            total_value += value

            # Menampilkan nomor/nilai koin pada gambar
            cv2.putText(
                output_img, 
                f"#{i+1}: Rp{value}", 
                (x - 40, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )

    return output_img, total_coins, total_value

# Komponen Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar koin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar yang diunggah
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Konversi RGB ke BGR untuk OpenCV jika diperlukan
    if image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    # Deteksi Koin
    result_img_bgr, count, value = process_and_count_coins(
        image_cv, min_dist, param1, param2, min_radius, max_radius
    )
    
    # Konversi kembali BGR ke RGB untuk ditampilkan di Streamlit
    result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

    # Tampilan Layout Dua Kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi Koin")
        st.image(result_img_rgb, use_container_width=True)

    st.markdown("---")
    
    # Ringkasan Hasil
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Total Koin Terdeteksi", value=f"{count} Koin")
    with res_col2:
        st.metric(label="Total Nilai Uang", value=f"Rp {value:,}")