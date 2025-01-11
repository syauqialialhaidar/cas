from flask import Flask, render_template, jsonify, request, session
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from transformers import pipeline
import pickle
from config import MYSQL_CONFIG, APP_SECRET_KEY
from flask_mysqldb import MySQL
from flask import Flask, render_template, request, jsonify
from werkzeug.security import generate_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import check_password_hash
import smtplib
from langchain_community.vectorstores import FAISS  
from email.mime.text import MIMEText
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import pickle
import torch 
from flask import session
from flask import render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain   
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from datetime import datetime
from indobert import SentimentAnalyzer


app = Flask(__name__)

model_indobert = 'senti'
analyzer_indobert = SentimentAnalyzer(model_indobert)

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = MYSQL_CONFIG['host']
app.config['MYSQL_USER'] = MYSQL_CONFIG['user']
app.config['MYSQL_PASSWORD'] = MYSQL_CONFIG['password']
app.config['MYSQL_DB'] = MYSQL_CONFIG['db']

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfigurasi Flask
app.secret_key = APP_SECRET_KEY

groq_api_key = "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"
file_path = os.path.join(os.getcwd(), 'model_save', 'chatbot', 'vectorstore.pkl')

# Inisialisasi MySQL
mysql = MySQL(app)

MODEL_PATH = os.path.join('model', 'modelku.h5')  # Sesuaikan dengan lokasi model Anda
try:
    model = load_model(MODEL_PATH) # Debugging untuk memastikan model berhasil dimuat
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Daftar kelas yang digunakan oleh model gambar
class_names = ['coklat kehitaman', 'sawo matang', 'kuning langsat', 'putih']


# Route untuk Beranda
@app.route('/beranda')
def beranda():
    return render_template('beranda.html')
 
# Route untuk Chatbot
def initialize_llm(groq_api_key):
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    return llm

def initialize_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings

def create_rag_chain(retriever, llm):
    system_prompt = (
        "Anda adalah asisten untuk tugas menjawab pertanyaan yang bernama gold. "
        "Menjawab menggunakan bahasa indonesia "
        "Jika Anda tidak ada jawaban pada konteks, katakan saja menurut saya dan berikan jawaban yang sesuai "
        ". Gunakan maksimal empat kalimat dan pertahankan "
        "jawaban singkat.\n\n"
        "{context}"
    )

    retrieval_qa_chain = (
        {"context": retriever, "question": RunnablePassthrough() }
        | ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
        | llm
        | StrOutputParser()
    )
    return retrieval_qa_chain

def save_model(vectorstore, embeddings, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    vectorstore_path = os.path.join(save_dir, "chatbot/vectorstore.pkl")
    with open(vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)

    embeddings_path = os.path.join(save_dir, "chatbot/embeddings.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)

@app.route('/chatbot', methods=['GET'])
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['POST'])
def chat():
    try:
        logging.debug("Request received at /chatbot endpoint.")
        data = request.get_json()
        logging.debug(f"Data received: {data}")

        user_input = data.get("message")
        if not user_input:
            logging.error("User input is missing.")
            return jsonify({"error": "Message is required"}), 400

        pdf_path = "C:/xampp/htdocs/project/data/datasetchatbot.pdf"
        groq_api_key = "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"
        logging.debug("Initializing LLM and embeddings...")

        llm = initialize_llm(groq_api_key)
        embeddings = initialize_embeddings()

        logging.debug("Loading PDF...")
        pdf_loader = PyPDFLoader(pdf_path)
        documents = pdf_loader.load()

        logging.debug("Creating vector store...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        logging.debug("Saving model and embeddings...")
        save_model(vectorstore, embeddings, save_dir="model_save")

        logging.debug("Creating RAG chain...")
        rag_chain = create_rag_chain(retriever, llm)

        logging.debug("Invoking RAG chain...")
        response = rag_chain.invoke(user_input)

        logging.debug(f"Response generated: {response}")
        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    


# Route untuk Deteksi
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
                logging.debug(f"Saving uploaded file to {filepath}.")
                file.save(filepath)

                # Proses gambar
                logging.debug("Processing image...")
                img = load_img(filepath, target_size=(224, 224))  # Sesuaikan ukuran input model
                img_array = img_to_array(img) / 255.0  # Normalisasi
                img_array = np.expand_dims(img_array, axis=0)
                logging.debug(f"Image shape before prediction: {img_array.shape}")

                # Prediksi dengan model
                logging.debug("Making predictions...")
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
                logging.error(f"Error during image processing: {str(e)}", exc_info=True)
                return jsonify({'error': str(e)}), 500
    return render_template('deteksi.html')



# Route untuk Profil
@app.route('/profil')
def profil():
    # Cek apakah ada sesi pengguna yang sedang login
    if 'user_id' not in session:
        flash('Anda harus login terlebih dahulu!', 'warning')
        return redirect(url_for('login'))  # Redirect ke halaman login jika pengguna belum login

    # Ambil data pengguna berdasarkan user_id yang disimpan di sesi
    user_id = session['user_id']
    cur = mysql.connection.cursor()
    cur.execute("SELECT username, email FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()

    if user:
        # Kirim data pengguna ke template
        return render_template('profile.html', username=user[0], email=user[1])
    else:
        flash('Pengguna tidak ditemukan.', 'danger')
        return redirect(url_for('login'))


# Route untuk Outfit
@app.route('/outfit')
def outfit():
    return render_template('outfit.html')

@app.route('/current_user', methods=['GET'])
def current_user():
    # Memeriksa apakah pengguna sudah login
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    # Ambil informasi pengguna dari sesi
    user_id = session['user_id']
    username = session['username']
    email = session['email']

    return jsonify({
        'user_id': user_id,
        'username': username,
        'email': email
    })


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

        if user and check_password_hash(user[3], password):  # Pastikan indeks password sesuai
            # Simpan informasi pengguna di sesi
            session['user_id'] = user[0]  # ID pengguna
            session['username'] = user[1]  # Nama pengguna
            session['email'] = user[2]  # Email pengguna
            role = user[4]  # Pastikan indeks role sesuai dengan posisi kolom role di tabel
            flash('Login berhasil!', 'success')

            if role == 'admin':
                return redirect(url_for('dashboard'))  # Ganti dengan route dashboard admin Anda
            else:
                return redirect(url_for('beranda'))  # Ganti dengan route beranda user Anda
        else:
            flash('Email atau password salah.', 'danger')

    return render_template('login.html')


# Route untuk Registrasi
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Ambil data dari form registrasi
        username = request.form.get('username') 
        email = request.form.get('email')
        password = request.form.get('password')

        # Validasi jika data tidak lengkap
        if not username or not email or not password:
            flash('Semua field harus diisi!', 'danger')
            return render_template('register.html') 

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256') 

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
            server.login(from_email, 'your_email_password')  
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
    
    
@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        flash('Anda harus login terlebih dahulu!', 'warning')
        return redirect(url_for('login'))  # Redirect ke halaman login jika pengguna belum login

    user_id = session['user_id']

    # Ambil data pengguna berdasarkan user_id yang disimpan di sesi
    cur = mysql.connection.cursor()
    cur.execute("SELECT username, email FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()

    if request.method == 'POST':
        new_username = request.form['username']
        new_email = request.form['email']

        if not new_username or not new_email:
            flash('Semua field harus diisi!', 'danger')
            return render_template('profile.html', username=user[0], email=user[1])

        try:
            # Perbarui data pengguna
            cur = mysql.connection.cursor()
            cur.execute("UPDATE users SET username = %s, email = %s WHERE id = %s",
                        (new_username, new_email, user_id))
            mysql.connection.commit()
            cur.close()

            # Update sesi dengan data terbaru
            session['username'] = new_username
            session['email'] = new_email

            flash('Profil berhasil diperbarui!', 'success')
            return redirect(url_for('profil'))  # Redirect ke halaman profil setelah pembaruan
        except Exception as e:
            flash(f'Terjadi kesalahan: {str(e)}', 'danger')
            return render_template('profile.html', username=user[0], email=user[1])

    return render_template('profile.html', username=user[0], email=user[1])


@app.route('/dashboard')
def dashboard():
    return render_template('admin/dashboard.html')

@app.route('/profiladmin')
def profiladmin():
    return render_template('admin/profiladmin.html')

@app.route('/ulasan')
def ulasan():
    try:
        # Ambil data ulasan dari database
        cur = mysql.connection.cursor()
        cur.execute("SELECT text FROM reviews")
        reviews = cur.fetchall()  # Mengambil semua ulasan
        cur.close()

        sentiment_results = []
        for review in reviews:
            review_text = review[0]  # Karena hasilnya berupa tuple
            predicted_class, probabilities = analyzer_indobert.predict_sentiment(review_text)
            sentiment = "Positif" if predicted_class == 1 else "Negatif"
            sentiment_results.append({
                "text": review_text,
                "sentiment": sentiment
            })

        # Render hasil sentimen ke template
        return render_template('admin/ulasan.html', sentiment_results=sentiment_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route("/logout", methods=['GET', 'POST'])
def logout():
    # Hapus semua data sesi
    session.clear()
    # Flash pesan logout sukses
    flash('Anda telah berhasil logout.', 'info')
    # Redirect ke halaman login
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Pastikan folder 'uploads' ada untuk menyimpan file sementara
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
