import streamlit as st
import pickle
import numpy as np
import plotly.express as px
import pandas as pd

# تحميل النموذج والمعالج
dtr = pickle.load(open('dtr.pkl', 'rb'))
prepro = pickle.load(open('preprocessor.pkl', 'rb'))

# إعداد أنماط CSS لتحسين الشكل
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    .stSelectbox, .stNumberInput {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .title {
        color: #1b5e20;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        color: #424242;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# العنوان والصورة
st.markdown('<p class="title">نظام توقع إنتاجية المحاصيل الزراعية</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">تنبؤ ذكي بإنتاجية المحاصيل بناءً على البيانات البيئية</p>', unsafe_allow_html=True)

st.image(
    'https://fnb.tech/wp-content/uploads/2025/04/Best-Cover-Crops-you-Should-Plant.jpeg',
    caption="حقل زراعي مزدهر",
    use_column_width=True
)

# وصف التطبيق
st.markdown("أدخل بيانات المنطقة، نوع المحصول، والظروف البيئية لتوقع الإنتاجية بدقة.")

# قوائم المدخلات
items = ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat', 
         'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams']
area = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 
        'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 
        'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada', 
        'Central African Republic', 'Chile', 'Colombia', 'Croatia', 'Denmark', 
        'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea', 'Estonia', 
        'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 
        'Guyana', 'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Iraq', 
        'Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 
        'Lebanon', 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 
        'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 
        'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 
        'Niger', 'Norway', 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 
        'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia', 
        'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 
        'Switzerland', 'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 
        'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe']

# تنظيم المدخلات في أعمدة
st.header("أدخل البيانات")
col1, col2 = st.columns(2)

with col1:
    select_area = st.selectbox('اختر المنطقة', area, help="اختر الدولة أو المنطقة")
    select_item = st.selectbox('اختر نوع المحصول', items, help="اختر نوع المحصول")

with col2:
    year = st.number_input('السنة', max_value=2025, min_value=1990, value=1990, help="أدخل سنة الحصاد")
    avg_rain = st.number_input('متوسط هطول الأمطار (ملم/سنة)', min_value=0.0, value=1485.0, help="أدخل متوسط الأمطار السنوي")

col3, col4 = st.columns(2)
with col3:
    pesticides_tonnes = st.number_input('كمية المبيدات (طن)', min_value=0.0, value=121.0, help="أدخل كمية المبيدات المستخدمة")
with col4:
    avg_temp = st.number_input('متوسط درجة الحرارة (°م)', min_value=-50.0, max_value=50.0, value=16.37, help="أدخل متوسط درجة الحرارة")

# دالة التنبؤ
def prediction(Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp):
    feature = np.array([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])
    trans_feat = prepro.transform(feature)
    return dtr.predict(trans_feat).reshape(1, -1)

# زر التنبؤ
if st.button("توقع الإنتاجية", key="predict_button"):
    try:
        result = prediction(select_area, select_item, year, avg_rain, pesticides_tonnes, avg_temp)
        st.markdown(f'<p class="result">الإنتاجية المتوقعة لـ {select_item} في {select_area}: {result[0][0]:.2f} هكتوغرام/هكتار</p>', unsafe_allow_html=True)
        
        # رسم بياني للنتيجة
        df = pd.DataFrame({
            'Crop': [select_item],
            'Predicted Yield (hg/ha)': [result[0][0]]
        })
        fig = px.bar(df, x='Crop', y='Predicted Yield (hg/ha)', 
                     title=f"Predicted Yield for {select_item} in {select_area}",
                     color_discrete_sequence=['#2e7d32'])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")

# إضافة ملاحظات إضافية
st.markdown("---")
st.markdown("**ملاحظات**:")
st.markdown("- تأكد من إدخال بيانات دقيقة للحصول على توقعات موثوقة.")
st.markdown("- النموذج يعتمد على بيانات تاريخية من 1990 إلى 2013.")
st.markdown("- لمزيد من التفاصيل، تواصلوا مع فريق دعم الزراعة الذكية.")