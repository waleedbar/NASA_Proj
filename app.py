import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

# --- تهيئة تطبيق فلاسك ---
app = Flask(__name__)

# --- تحميل النموذج المحفوظ ---
MODEL_PATH = 'exoplanet_model_light.pkl'
model_package = None

try:
    model_package = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at '{MODEL_PATH}'. Make sure it's in the same folder as app.py.")
    model_package = None # استمر في تشغيل التطبيق لعرض الخطأ في الواجهة
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_package = None

# استخراج المكونات من الحزمة إذا تم تحميلها بنجاح
if model_package:
    model = model_package.get('model')
    model_feature_names = model_package.get('feature_names')
    class_names = model_package.get('class_names', ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])
else:
    model = None

# --- دالة لمعالجة البيانات الجديدة ---
def preprocess_new_data(df):
    """
    تطبيق نفس خطوات هندسة الميزات والمعالجة المسبقة على البيانات الجديدة
    """
    X_new = df.copy()

    # 1. هندسة الميزات (نفس الكود المستخدم في التدريب)
    with np.errstate(divide='ignore', invalid='ignore'):
        X_new['radius_ratio'] = X_new['planet_radius'] / X_new['stellar_radius']
        X_new['density_proxy'] = X_new['planet_radius'] / (X_new['orbital_period'] ** (2/3))
        X_new['insolation_flux'] = (X_new['stellar_temp'] / 5778) ** 4 * (X_new['stellar_radius'] ** 2) / (X_new['orbital_period'] ** (2/3))
        X_new['temp_gravity_interaction'] = X_new['stellar_temp'] * X_new['stellar_gravity']
        X_new['period_depth_ratio'] = X_new['orbital_period'] / (X_new['transit_depth'] + 1e-10)
        X_new['duration_period_ratio'] = X_new['transit_duration'] / X_new['orbital_period']

    X_new.replace([np.inf, -np.inf], np.nan, inplace=True)

    # التأكد من أن جميع الأعمدة المطلوبة موجودة
    for col in model_feature_names:
        if col not in X_new.columns:
            X_new[col] = np.nan # أو قيمة افتراضية مناسبة

    # إعادة ترتيب الأعمدة لتطابق ترتيبها أثناء التدريب
    X_new = X_new[model_feature_names]

    return X_new


# --- تعريف المسارات (Routes) ---
@app.route('/')
def home():
    """
    عرض الصفحة الرئيسية
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    استقبال البيانات، إجراء التنبؤ، وإرجاع النتائج
    """
    if not model:
        return jsonify({'error': 'Model is not loaded on the server. Please check the server logs.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and model:
        try:
            # قراءة ملف CSV
            df_new = pd.read_csv(file)
            
            # معالجة البيانات
            processed_df = preprocess_new_data(df_new)
            
            # إجراء التنبؤ
            predictions = model.predict(processed_df)
            
            # حساب الإحصائيات
            # تحويل التنبؤات الرقمية إلى أسماء الفئات
            prediction_labels = [class_names[p] for p in predictions]
            
            results = {
                'total_rows': len(prediction_labels),
                'confirmed_exoplanets': prediction_labels.count('CONFIRMED'),
                'planet_candidates': prediction_labels.count('CANDIDATE'),
                'false_positives': prediction_labels.count('FALSE POSITIVE')
            }

            return jsonify(results)

        except Exception as e:
            return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500
            
    return jsonify({'error': 'Model not loaded or file not provided'}), 400


# --- تشغيل التطبيق ---
if __name__ == '__main__':
    # استخدم '0.0.0.0' لجعل الخادم متاحًا على الشبكة
    app.run(host='0.0.0.0', port=5000, debug=True)