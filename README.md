# AI Skincare Companion

A web app that classifies skin type from a selfie using a fine-tuned InceptionV3 model and recommends skincare products with the help of a Groq-powered chatbot (LLM).

## 🚀 Features

* Upload a selfie to detect skin type (Acne, Dry, Oily, Normal)
* Get personalized product suggestions via LLM chat
* Built with Flask, TensorFlow, and Groq API

## 📦 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create a `.env` file with your credentials

After cloning, create a file named `.env` in the project folder and add your API key:

```env
GROQ_API_KEY=your-groq-api-key-here
```

You can sign up for a free API key at [https://console.groq.com/keys](https://console.groq.com/keys).


### 3. Install dependencies

Make sure you have Python 3.9+ installed. Then run:

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
python app.py
```

Visit `http://localhost:8000` in your browser to try it out!

## 🧠 Model Info

* Transfer learning with InceptionV3
* Weights are loaded from `best_finetuned_InceptionV3.h5`

## 📁 Folder Structure

```
skin-diagnostic/
├── app.py
├── model.ipynb
├── best_finetuned_InceptionV3.h5
├── .env              
├── .gitignore
├── requirements.txt
└── README.md
```

## 📜 License

MIT License. See `LICENSE` file for details.
