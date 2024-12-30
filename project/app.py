from flask import Flask, render_template, jsonify, request, session
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from transformers import pipeline
from config import MYSQL_CONFIG, APP_SECRET_KEY
from flask_mysqldb import MySQL
from flask import Flask, render_template, request, jsonify
from werkzeug.security import generate_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import check_password_hash
import smtplib
from email.mime.text import MIMEText



app = Flask(__name__)

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = MYSQL_CONFIG['host']
app.config['MYSQL_USER'] = MYSQL_CONFIG['user']
app.config['MYSQL_PASSWORD'] = MYSQL_CONFIG['password']
app.config['MYSQL_DB'] = MYSQL_CONFIG['db']

# Konfigurasi Flask
app.secret_key = APP_SECRET_KEY

# Inisialisasi MySQL
mysql = MySQL(app)

# Muat model gambar
MODEL_PATH = os.path.join('model', 'modelku.h5')  # Sesuaikan dengan lokasi model Anda
try:
    model = load_model(MODEL_PATH) # Debugging untuk memastikan model berhasil dimuat
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Daftar kelas yang digunakan oleh model gambar
class_names = ['coklat kehitaman', 'sawo matang', 'kuning langsat']

# Route untuk Beranda
@app.route('/beranda')
def beranda():
    return render_template('beranda.html')

# Route untuk Chatbot
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Route untuk Deteksi, mendukung GET dan POST
@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nama file kosong'}), 400
        if file:
            filepath = os.path.join('uploads', file.filename)
            try:
                # Simpan file sementara
                file.save(filepath)

                # Proses gambar
                img = load_img(filepath, target_size=(150, 150))  # Sesuaikan ukuran input model
                img_array = img_to_array(img) / 255.0  # Normalisasi
                img_array = np.expand_dims(img_array, axis=0)

                # Prediksi dengan model
                if model:
                    predictions = model.predict(img_array)
                    class_index = np.argmax(predictions[0])
                    result = class_names[class_index]
                else:
                    result = "Model gambar tidak tersedia"

                # Hapus file setelah selesai
                os.remove(filepath)

                return jsonify({
                    'predicted_class': result,
                    'predictions': predictions[0].tolist() if model else []
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('deteksi.html')

# Route untuk Profil
@app.route('/profile')
def profil():
    return render_template('profile.html')

# Route untuk Outfit
@app.route('/outfit')
def outfit():
    return render_template('outfit.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        # Ambil data dari database
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, email FROM users")  # Pilih kolom yang ingin ditampilkan
        users = cur.fetchall()
        cur.close()

        # Konversi data ke dalam format JSON
        users_list = [
            {
                'id': user[0],
                'username': user[1],
                'email': user[2]
            } for user in users
        ]

        return jsonify({
            'status': 'success',
            'data': users_list
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



# Route untuk Login
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Cek apakah email ada di database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user and check_password_hash(user[3], password):  # Misalkan password ada di index 3
            flash('Login berhasil!', 'success')
            return redirect(url_for('beranda'))  # Ganti dengan halaman yang sesuai
        else:
            flash('Email atau password salah.', 'danger')
    
    return render_template('login.html')

# Route untuk Registrasi
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Ambil data dari form registrasi
        username = request.form.get('username')  # Menggunakan get untuk menghindari KeyError
        email = request.form.get('email')
        password = request.form.get('password')

        # Validasi jika data tidak lengkap
        if not username or not email or not password:
            flash('Semua field harus diisi!', 'danger')
            return render_template('register.html')  # Mengembalikan form registrasi jika ada field yang kosong

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')  # Menggunakan pbkdf2:sha256

        # Periksa apakah email sudah ada di database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            flash('Email sudah terdaftar!', 'danger')
            return render_template('register.html')  

        cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        mysql.connection.commit()
        cur.close()

        flash('Akun berhasil dibuat! Silakan login.', 'success')
        return redirect(url_for('login')) 

    return render_template('register.html') 

# Route untuk Forgot Password
@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']

        # Cek apakah email dan nama cocok dengan yang ada di database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s AND name = %s", (email, name))
        user = cur.fetchone()
        cur.close()

        if user:
            send_reset_email(user[2])  # Misalnya email ada di user[2]
            flash('Kami telah mengirimkan email untuk mereset password Anda!', 'success')
        else:
            flash('Email atau nama tidak ditemukan.', 'danger')

    return render_template('forgot.html')

def send_reset_email(to_email):
    # Pengaturan pengiriman email untuk reset password
    from_email = "your_email@example.com"
    subject = "Reset Password"
    body = "Klik link ini untuk mereset password Anda: http://localhost:5000/reset-password"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Mengirimkan email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(from_email, 'your_email_password')  # Masukkan password email Anda
            server.sendmail(from_email, to_email, msg.as_string())
            print("Email terkirim")
    except Exception as e:
        print(f"Error: {e}")

@app.route('/add_review', methods=['POST'])
def add_review():
    data = request.get_json()
    name = data.get('name')
    text = data.get('text')

    if not name or not text:
        return jsonify({'error': 'Name and text are required'}), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO reviews (name, text) VALUES (%s, %s)", (name, text))
        mysql.connection.commit()
        cur.close()

        return jsonify({'status': 'success', 'name': name, 'text': text}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Pastikan folder 'uploads' ada untuk menyimpan file sementara
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
