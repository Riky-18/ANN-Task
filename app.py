import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title='VinoScan | Red Wine Quality Intelligence',
    page_icon='🍷',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 15%, rgba(110, 20, 55, 0.45) 0%, rgba(20, 12, 32, 0.95) 75%, rgba(10, 5, 18, 1) 100%);
    color: #f8f9fa;
}

.glass-hero {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 20px;
    padding: 26px 30px;
    margin-bottom: 22px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4);
}

.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: rgba(255, 255, 255, 0.72);
    margin-bottom: 14px;
}

.tag-badge {
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 999px;
    padding: 6px 16px;
    display: inline-block;
    font-size: 0.82rem;
    margin-right: 8px;
    color: #ffffff;
}

.verdict-high {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.22) 0%, rgba(39, 174, 96, 0.12) 100%);
    border: 1px solid #2ecc71;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}

.verdict-standard {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.22) 0%, rgba(192, 57, 43, 0.12) 100%);
    border: 1px solid #e74c3c;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_assets():
    return joblib.load('artifacts/wine_quality_model.joblib')

if not os.path.exists('artifacts/wine_quality_model.joblib'):
    st.error('Model artifacts not found! Please run python pipeline.py first.')
    st.stop()

bundle = load_model_assets()
model = bundle['model']
feature_cols = bundle['features']
model_accuracy = bundle['accuracy']
model_auc = bundle['auc']

st.markdown(f"""
<div class="glass-hero">
    <div class="hero-title">🍷 VinoScan — Wine Quality AI Diagnostics</div>
    <div class="hero-subtitle">Instantly predict whether a red wine is High Quality (Grade ≥ 7) or Standard Quality based on its chemical profile.</div>
    <div>
        <span class="tag-badge">Accuracy: <b>{model_accuracy * 100:.1f}%</b></span>
        <span class="tag-badge">ROC-AUC: <b>{model_auc:.3f}</b></span>
        <span class="tag-badge">Model: <b>Balanced Random Forest</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(['🔬 Single Sample Tester', '📈 Exploratory Data Charts'])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader('Step 1: Set Physicochemical Wine Properties')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('**Acidity Profile**')
        fixed_acidity = st.slider('Fixed Acidity (Tartaric Acid) (g/dm³)', 4.0, 16.0, 8.3, 0.1)
        volatile_acidity = st.slider('Volatile Acidity (Acetic Acid) (g/dm³)', 0.10, 1.60, 0.52, 0.01)
        citric_acid = st.slider('Citric Acid (g/dm³)', 0.0, 1.0, 0.27, 0.01)
        pH = st.slider('pH Level', 2.7, 4.1, 3.31, 0.01)

    with col2:
        st.write('**Fermentation & Body**')
        alcohol = st.slider('Alcohol (% by Volume)', 8.0, 15.0, 10.4, 0.1)
        residual_sugar = st.slider('Residual Sugar (g/dm³)', 0.5, 15.5, 2.5, 0.1)
        density = st.slider('Density (g/cm³)', 0.9900, 1.0040, 0.9967, 0.0002, format='%.4f')
        chlorides = st.slider('Chlorides (Salt Content) (g/dm³)', 0.01, 0.60, 0.08, 0.005)

    with col3:
        st.write('**Preservatives & Sulphates**')
        sulphates = st.slider('Sulphates (g/dm³)', 0.3, 2.0, 0.65, 0.02)
        free_so2 = st.slider('Free Sulfur Dioxide (mg/dm³)', 1.0, 72.0, 14.0, 1.0)
        total_so2 = st.slider('Total Sulfur Dioxide (mg/dm³)', 6.0, 289.0, 46.0, 1.0)

    total_acidity = fixed_acidity + volatile_acidity + citric_acid
    free_sulfur_ratio = free_so2 / (total_so2 + 1e-5)
    alcohol_to_sugar_ratio = alcohol / (residual_sugar + 1e-5)

    input_df = pd.DataFrame([{
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_so2,
        'total sulfur dioxide': total_so2,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'total_acidity': total_acidity,
        'free_sulfur_ratio': free_sulfur_ratio,
        'alcohol_to_sugar_ratio': alcohol_to_sugar_ratio
    }])[feature_cols]

    st.markdown('---')

    if st.button('Analyze Wine Sample', type='primary', use_container_width=True):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        good_prob = probabilities[1] * 100
        standard_prob = probabilities[0] * 100

        res_col1, res_col2 = st.columns([1.1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="verdict-high">
                    <h2 style="color: #2ecc71; margin-bottom: 4px;">🏅 Premium Quality Wine</h2>
                    <p style="font-size: 0.95rem; color: #ecf0f1;">This wine sample matches the benchmark chemistry for top-rated wines (Rating ≥ 7).</p>
                    <h1 style="font-size: 2.8rem; margin: 8px 0; color: #ffffff;">{good_prob:.1f}%</h1>
                    <span style="font-size: 0.85rem; color: #bdc3c7;">Confidence of Premium Tier</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-standard">
                    <h2 style="color: #e74c3c; margin-bottom: 4px;">🍷 Standard Table Wine</h2>
                    <p style="font-size: 0.95rem; color: #ecf0f1;">This wine sample matches the chemistry of common table wines (Rating < 7).</p>
                    <h1 style="font-size: 2.8rem; margin: 8px 0; color: #ffffff;">{standard_prob:.1f}%</h1>
                    <span style="font-size: 0.85rem; color: #bdc3c7;">Confidence of Standard Tier</span>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            fig, ax = plt.subplots(figsize=(5, 2.7))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')

            labels = ['Standard', 'Premium']
            values = [standard_prob, good_prob]
            bar_colors = ['#e74c3c', '#2ecc71']

            bars = ax.barh(labels, values, color=bar_colors, height=0.45)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability Score (%)', color='#ffffff', fontsize=9)
            ax.tick_params(colors='#ffffff', labelsize=9)

            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.2))

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 2, bar.get_y() + 0.14, f'{w:.1f}%', color='white', fontweight='bold', fontsize=9)

            st.pyplot(fig, transparent=True)
            plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader('Exploratory Data Analysis & Performance Atlas')

    chart_selection = st.selectbox(
        'Select visualization to view:',
        [
            ('1. Quality Target Stratification', 'plots/01_quality_distribution.png'),
            ('2. Correlation Heatmap', 'plots/02_correlation_matrix.png'),
            ('3. Key Features Boxplot (Alcohol & Volatile Acidity)', 'plots/03_key_features_comparison.png'),
            ('4. Confusion Matrix & ROC Curve', 'plots/04_model_evaluation.png')
        ],
        format_func=lambda x: x[0]
    )

    if os.path.exists(chart_selection[1]):
        st.image(chart_selection[1], use_container_width=True)
    else:
        st.info('Charts have not been generated yet. Please run python pipeline.py first.')

    st.markdown('</div>', unsafe_allow_html=True)