import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from component.nav import navbar
from component.bootstrap import bootstrap


# ------------------navbar-----------------------
navbar()
def deskPre() :
    bootstrap()
    desk_pre = """
    <p>
        Preprocessing data merupakan tahap yang sangat penting sebelum data diproses atau diolah dalam Machine Learning (ML). Proses preprocessing bertujuan untuk mempersiapkan data agar dapat diolah dengan baik oleh sistem. Tahapan preprocessing yang dilakukan dalam penelitian ini meliputi penanganan outlier, label encoding, dan normalisasi.
    </p>
    <br>
    <div class="list-group">
        <ol class="list-group list-group-numbered">
            <li class="list-group-item">
            <strong>Penanganan Outlier:</strong>
            Mengidentifikasi dan menangani outlier atau nilai yang tidak biasa dalam dataset yang dapat mempengaruhi performa model.
            </li>
            <li class="list-group-item">
            <strong>Label Encoding:</strong>
            Mengonversi label kategori menjadi representasi numerik agar dapat diproses oleh algoritma.</li>
            <li class="list-group-item">
            <strong>Normalisasi:</strong>
            Melakukan transformasi data agar memiliki skala yang sama, misalnya semua fitur berada dalam rentang nilai tertentu.</li>
        </ol>
    </div>
    """
    st.markdown(desk_pre,unsafe_allow_html=True)
def pre():
    bootstrap()
    # ------------------FULL DATA-----------------------
    maternal_health_risk = fetch_ucirepo(id=863)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)
    maternal_health_risk = pd.concat([X_df, y_df], axis=1)

    # Rename columns for better readability
    maternal_health_risk.rename(columns={
        'BS': 'BloodSugar'
    }, inplace=True)

    numerical_columns = [col for col in maternal_health_risk.columns if maternal_health_risk[col].dtype != 'O']
    categorical_columns = [col for col in maternal_health_risk.columns if maternal_health_risk[col].dtype == 'O']
    discrete_columns = [col for col in maternal_health_risk.columns if len(maternal_health_risk[col].unique())<7]
    continuous_columns = [col for col in maternal_health_risk.columns if len(maternal_health_risk[col].unique())>10]

    df_main = maternal_health_risk.copy()

    # Tampilkan hasil ke halaman Streamlit
    st.write("### Data sebelum dilakukan Label Encoder")
    st.write(df_main)

    # Tampilkan hasil ke halaman Streamlit
    st.write("### Data setelah dilakukan Label Encoder")
    for col in categorical_columns:
        le = LabelEncoder()
        df_main[col] = le.fit_transform(df_main[col])

    st.write(df_main)

    # OUTLIER
    df_main_lof = df_main.copy()
    # Menggunakan semua kolom fitur untuk deteksi outlier
    columns_for_lof = list(X.columns)
    # Inisialisasi model LOF
    lof = LocalOutlierFactor(n_neighbors=40, p=2)
    # Mendeteksi outlier
    outlier_scores = lof.fit_predict(X)
    df_main_lof['LOF_Score'] = -lof.negative_outlier_factor_
    # Menampilkan DataFrame dengan skor LOF
    df_main_lof.head(len(df_main_lof))
    df_main_lof.insert(0, 'ID', df_main_lof.index)
    print("Outlier data:")
    for i, row in df_main_lof.iterrows():
        if outlier_scores[i] == -1:
            print(f"ID {row['ID']}, LOF Score: {row['LOF_Score']:.2f}")
    data_outlier = df_main_lof.loc[df_main_lof['LOF_Score'] > 2]
    st.write("### Data outlier:")
    st.write(data_outlier)

    df_main_lof = df_main_lof.loc[df_main_lof['LOF_Score'] <= 2]
    st.write("### Jumlah data setelah penanganan outlier:")
    st.write(df_main_lof)

    # Normalisasi data
    X_train_model = df_main_lof.drop(columns=['ID','RiskLevel','LOF_Score'])
    Y_train_model = df_main_lof['RiskLevel']
    features = pd.DataFrame(X_train_model).columns
    target = pd.DataFrame(Y_train_model).columns
    df_prep = df_main_lof[features]
    scaler = MinMaxScaler()
    x = scaler.fit_transform(df_prep)
    y = df_main_lof[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)


# ------------------PreProcessing Data-----------------------
st.markdown(
    '''
    <h1 align="center">PRE PROCESSING DATA</h1>
    '''
    , unsafe_allow_html=True
)
deskPre()
st.subheader('PreProcessing')
st.code("""
    import streamlit as st
    import pandas as pd
    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from component.nav import navbar
    from component.bootstrap import bootstrap
    
    maternal_health_risk = fetch_ucirepo(id=863)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)
    maternal_health_risk = pd.concat([X_df, y_df], axis=1)

    # Rename columns for better readability
    maternal_health_risk.rename(columns={
        'BS': 'BloodSugar'
    }, inplace=True)

    numerical_columns = [col for col in maternal_health_risk.columns if maternal_health_risk[col].dtype != 'O']
    categorical_columns = [col for col in maternal_health_risk.columns if maternal_health_risk[col].dtype == 'O']
    discrete_columns = [col for col in maternal_health_risk.columns if len(maternal_health_risk[col].unique())<7]
    continuous_columns = [col for col in maternal_health_risk.columns if len(maternal_health_risk[col].unique())>10]

    df_main = maternal_health_risk.copy()

    # Tampilkan hasil ke halaman Streamlit
    st.write("### Data sebelum dilakukan Label Encoder")
    st.write(df_main)

    # Tampilkan hasil ke halaman Streamlit
    st.write("### Data setelah dilakukan Label Encoder")
    for col in categorical_columns:
        le = LabelEncoder()
        df_main[col] = le.fit_transform(df_main[col])

    st.write(df_main)

    # OUTLIER
    df_main_lof = df_main.copy()
    # Menggunakan semua kolom fitur untuk deteksi outlier
    columns_for_lof = list(X.columns)
    # Inisialisasi model LOF
    lof = LocalOutlierFactor(n_neighbors=40, p=2)
    # Mendeteksi outlier
    outlier_scores = lof.fit_predict(X)
    df_main_lof['LOF_Score'] = -lof.negative_outlier_factor_
    # Menampilkan DataFrame dengan skor LOF
    df_main_lof.head(len(df_main_lof))
    df_main_lof.insert(0, 'ID', df_main_lof.index)
    print("Outlier data:")
    for i, row in df_main_lof.iterrows():
        if outlier_scores[i] == -1:
            print(f"ID {row['ID']}, LOF Score: {row['LOF_Score']:.2f}")
    data_outlier = df_main_lof.loc[df_main_lof['LOF_Score'] > 2]
    st.write("### Data outlier:")
    st.write(data_outlier)

    df_main_lof = df_main_lof.loc[df_main_lof['LOF_Score'] <= 2]
    st.write("### Jumlah data setelah penanganan outlier:")
    st.write(df_main_lof)

    # Normalisasi data
    X_train_model = df_main_lof.drop(columns=['ID','RiskLevel','LOF_Score'])
    Y_train_model = df_main_lof['RiskLevel']
    features = pd.DataFrame(X_train_model).columns
    target = pd.DataFrame(Y_train_model).columns
    df_prep = df_main_lof[features]
    scaler = MinMaxScaler()
    x = scaler.fit_transform(df_prep)
    y = df_main_lof[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
    """)
pre()
