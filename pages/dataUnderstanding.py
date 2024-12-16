import streamlit as st
import numpy as np
import pandas as pd
from component.bootstrap import bootstrap
from component.nav import navbar
from ucimlrepo import fetch_ucirepo

# def dataUnderstanding():
# ------------------navbar-----------------------
navbar()

def html():
    bootstrap()
    st.header('Deskripsi Data')
    st.subheader ('Maternal Health Risk')
    deskripsi = f"""
    <div class="container">
        <div class="card mt-3">
            <div class="card-body">
                Dataset Maternal Health Risk yang tersedia di UCIML Repository adalah sebuah dataset yang digunakan untuk mengklasifikasikan risiko kesehatan ibu hamil berdasarkan berbagai faktor atau fitur yang diukur selama kehamilan. Risiko kesehatan ibu hamil ini dikelompokkan dalam tiga kategori: High Risk (Risiko Tinggi), Mid Risk (Risiko Sedang), dan Low Risk (Risiko Rendah). Klasifikasi ini bertujuan untuk membantu profesional medis dalam melakukan prediksi dan intervensi dini untuk ibu hamil berdasarkan faktor-faktor kesehatan yang relevan.
            </div>
        </div>
        <br>
        <div class="list-group">
            <p>Berikut adalah penjelasan mengenai fitur-fitur yang digunakan dalam dataset ini:</strong></p>
            <ol class="list-group list-group-numbered">
                <li class="list-group-item">Age</li>
                <li class="list-group-item">SystolicBP</li>
                <li class="list-group-item">DiastolicBP</li>
                <li class="list-group-item">BloodSugar</li>
                <li class="list-group-item">BodyTemp</li>
                <li class="list-group-item">HeartRate</li>
            </ol>
        </div>
    </div>
    """
    st.markdown(deskripsi, unsafe_allow_html=True)
    st.image('asset/bumil.jpg',caption='Maternal Health Risk', width=500)

def desk_fitur():
    bootstrap()
    fitur = """
    <p>Berikut penjelasan tentang setiap fitur: </p>

    <div class="list-group">
        <ol class="list-group list-group-numbered">
            <li class="list-group-item"><strong>Age</strong> merujuk pada usia ibu hamil pada saat data dikumpulkan. Usia adalah salah satu faktor penting dalam menentukan risiko kesehatan ibu selama kehamilan. Secara umum, ibu hamil yang berusia sangat muda (di bawah 18 tahun) atau lebih tua (di atas 35 tahun) cenderung memiliki risiko kesehatan yang lebih tinggi dibandingkan ibu hamil yang berada dalam rentang usia 18-35 tahun. Data usia biasanya diambil langsung dari rekam medis atau wawancara dengan pasien.</li>
            <li class="list-group-item"><strong>Systolic BP</strong> yaitu tekanan darah pada saat jantung berkontraksi dan memompa darah ke seluruh tubuh. Tekanan darah sistolik yang tinggi dapat menjadi tanda hipertensi, yang berisiko menyebabkan komplikasi serius seperti preeklamsia, yang dapat membahayakan ibu dan bayi. Data ini diperoleh melalui pengukuran tekanan darah menggunakan alat tensimeter. Biasanya, pengukuran dilakukan oleh tenaga medis di klinik atau rumah sakit pada setiap kunjungan pemeriksaan kehamilan.
            </li>
            <li class="list-group-item"><strong>residual_sugar</strong> : Jumlah gula yang tersisa setelah fermentasi berhenti</li>
            <li class="list-group-item"><strong>Diastolic Blood Pressure (DiastolicBP)</strong> adalah tekanan darah diastolik, yaitu tekanan darah pada saat jantung berada dalam keadaan rileks antara dua detak. Tekanan darah diastolik yang tinggi juga dapat menunjukkan adanya risiko hipertensi dan berhubungan dengan kondisi-kondisi berbahaya lainnya selama kehamilan. Pengukuran tekanan darah diastolik dilakukan bersamaan dengan pengukuran tekanan darah sistolik menggunakan tensimeter. Seperti halnya systolic BP, diastolic BP diukur selama pemeriksaan rutin di fasilitas kesehatan.</li>
            <li class="list-group-item"><strong>Blood Sugar (BS)</strong> atau kadar gula darah mengacu pada kadar glukosa dalam darah ibu hamil. Kadar gula darah yang tinggi atau rendah dapat menjadi indikator risiko komplikasi kehamilan seperti diabetes gestasional atau hipoglikemia. Penting untuk memonitor kadar gula darah secara teratur pada ibu hamil untuk menghindari masalah kesehatan yang lebih besar. Data ini diukur melalui tes darah, yang dapat dilakukan di laboratorium atau menggunakan alat pengukur gula darah yang tersedia di klinik atau rumah sakit. 
            </li>
            <li class="list-group-item"><strong>Body Temperature (BodyTemp)</strong> adalah salah satu parameter penting yang dapat mencerminkan adanya infeksi atau gangguan lain dalam tubuh ibu hamil. Suhu tubuh yang tinggi bisa menjadi tanda adanya infeksi atau kondisi medis yang memerlukan perhatian medis segera. Suhu tubuh ibu hamil diukur menggunakan termometer, baik termometer digital, termometer inframerah, atau termometer air raksa pada setiap pemeriksaan rutin.</li>
        </ol>
    </div>
    """
    st.markdown(fitur, unsafe_allow_html=True)

def desk_kelas():
    bootstrap()
    kelas = """
     <div class="list-group">
            <p>Terdapat 3 kelas dari dataset yang digunakan, yaitu :</p>
            <ol class="list-group list-group-numbered">
                <li class="list-group-item">Low Risk</li>
                <li class="list-group-item">Mid Risk</li>
                <li class="list-group-item">High Risk</li>
            </ol>
        </div>
    """
    st.markdown(kelas, unsafe_allow_html=True)
    
    
def missval():
    misval="""
    <p>
    Disini kita dapat mengetahui apakah data set yang kita gunakan memiliki missing value atau tidak.
    </p>
    """
    st.markdown(misval, unsafe_allow_html=True)

def penjelasan_missval():
    penjelasan="""
    <p>
    dari hasil diatas kita dapat mengetahui bahwa dataset yang kita gunakan tidak memiliki missing value.
    </p>
    """
    st.markdown(penjelasan, unsafe_allow_html=True)
    
def display():
    st.markdown(
        '<h1 align="center">DATA UNDERSTANDING</h1>'
        ,unsafe_allow_html=True
    )
    html()
    # ------------------FULL DATA-----------------------
    st.subheader ('Data Maternal Health Risk')
    maternal_health_risk = fetch_ucirepo(id=863)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)
    maternal_health_risk = pd.concat([X_df, y_df], axis=1)

    st.write(maternal_health_risk)

    # ------------------PENJELASAN FITUR-----------------------
    st.subheader('Feature')
    'Jenis data dari setiap kolom:'
    dtypes = pd.DataFrame(maternal_health_risk.dtypes, columns=["Tipe Data"])
    st.dataframe(dtypes)

    desk_fitur()

    # ------------------PENJELASAN CLASS-----------------------
    st.subheader('Distribusi Kelas')
    y = np.array(y).flatten()
    class_counts = pd.Series(y).value_counts()
    st.write(class_counts)

    desk_kelas()

    st.subheader('Cek Missing Value')
    missval()
    maternal_health_risk_mv = maternal_health_risk.isnull().sum()
    maternal_health_risk_mv
    penjelasan_missval()
display()
