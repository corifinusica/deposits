from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import universal_forecast  
from werkzeug.utils import secure_filename
from waitress import serve


app = Flask(__name__)

# Конфигурация
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_request(req, res):
    with open('vsearch.log', 'a') as log:
        print(req.form, req.remote_addr, req.user_agent, res, file=log, sep='|')

@app.route('/')
@app.route('/entry')
def entry_page():
    return render_template('entry.html',
                           the_title='Welcome to search4letters on the web!')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Файл не найден', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'Файл не выбран', 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Предполагается, что universal_forecast.process_uploaded_file - синхронная функция
            results, error = universal_forecast.process_uploaded_file(filepath)
            
            if error:
                return render_template('error.html', 
                                     error=error,
                                     filename=filename)
            
            log_request(request, f"File processed: {filename}")
            
            return render_template('results.html', 
                                 results=results,
                                 filename=filename)
            
        except Exception as e:
            return render_template('error.html', 
                                 error=str(e),
                                 filename=filename)
    
    return 'Недопустимый формат файла', 400

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание результирующих файлов"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return 'Файл не найден', 404
    except Exception as e:
        return str(e), 500

@app.route('/health')
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000) 


