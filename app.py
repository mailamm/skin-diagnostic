import io, os, base64
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, request, session, render_template_string
from groq import Groq
import markdown
from dotenv import load_dotenv

# load .env variables
load_dotenv()

# —– Initialize Groq client —–
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# —– Configuration —–
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Acne","Dry","Oil","Normal"]
MODEL_PATH  = os.path.join(os.path.dirname(__file__),
                           "best_finetuned_InceptionV3.h5")
CHAT_MODEL  = "llama-3.3-70b-versatile"

# —– Build & load skin classification model —–
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def build_model(img_size=(224,224), n_classes=3):
    """
        Constructs a transfer-learning model based on InceptionV3,
        adds a global pooling layer + dropout + final dense classification.
    """

    # Load pre-trained InceptionV3 without top layers
    base = InceptionV3(include_top=False,
                          input_shape=(*img_size, 3),
                          weights="imagenet")
    base.trainable = False

    # Define new input and connect through base model
    inp = Input(shape=(*img_size, 3))
    x   = base(inp, training=False)
    x   = GlobalAveragePooling2D()(x)
    x   = Dropout(0.2)(x)
    out = Dense(n_classes, activation="softmax")(x)
    return Model(inp, out)


# Instantiate and load weights
skin_model = build_model(IMG_SIZE, len(CLASS_NAMES))
skin_model.load_weights(MODEL_PATH)

# —– Predict & return class + confidence —–
def predict_skin_type(image_bytes):
    """
        Given raw image bytes, preprocess and predict skin type + confidence.
    """
    # Load image, convert to RGB, resize
    img = (Image.open(io.BytesIO(image_bytes))
              .convert('RGB')
              .resize(IMG_SIZE))
    arr = np.array(img, dtype=np.float32)

    # Preprocess and batch
    x_pp = np.expand_dims(preprocess_input(arr), 0)
    preds = skin_model.predict(x_pp)[0]

    # Find top prediction and confidence
    idx = int(np.argmax(preds))
    conf = round(float(preds[idx]) * 100, 1)
    return CLASS_NAMES[idx], conf

# —– Chat wrappers —–
FEW_SHOT_CONTEXT = (
    # Examples to guide the model (not echoed back)
    "Example (do NOT repeat these):\n"
    "1. CeraVe Moisturizing Cream – Moisturizer: Rich, non-comedogenic hydration.\n"
    "2. La Roche-Posay Effaclar Duo – Treatment: Targets blemishes and unclogs pores.\n"
)

def chat_pipeline(prompt: str, model: str = CHAT_MODEL,
                  max_tokens: int = 400, temperature: float = 0.7,
                  top_p: float = 0.9, n: int = 1) -> list[dict]:
    """
        Send a prompt to the LLM and return a list of outputs with usage info.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user",  "content":prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n
    )
    out = []
    for choice in resp.choices:
        # Extract generated text and usage statistics
        txt = getattr(choice.message, 'content', str(choice.message))
        out.append({"generated_text": txt, "usage": resp.usage})
    return out

def chat_with_cot(prompt: str, **kwargs) -> str:
    """
        Chain-of-thought prompt wrapper: prepends a reasoning instruction.
    """
    cot_prompt = "Let’s think step by step:\n" + prompt
    return chat_pipeline(cot_prompt, **kwargs)[0]["generated_text"].strip()

# —– Flask application setup —–
app = Flask(__name__)
# Use a random secret key if none is provided
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))


# Inline HTML template (rendered via render_template_string)
HTML = '''
<!doctype html>
<html><head>
  <meta charset="utf-8">
  <title>Your AI Skincare Companion</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body {
      margin: 0; padding: 0;
      background: #f2f6f2;
      font-family: 'Poppins', sans-serif;
    }
    .container {
      width: 90%; max-width: 600px;
      margin: 30px auto;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.05);
      display: flex; flex-direction: column;
      overflow: hidden;
    }
    header {
      background: #5f9277;
      color: #ffffff;
      text-align: center;
      padding: 50px 20px;
      font-size: 2rem;
      font-weight: 500;
      font-weight: 700;  
    }
    .upload {
      padding: 20px;
      text-align: center;
      color: #475348;
      font-weight: 500;
    }
    .upload input[type=file] {
      margin-top: 10px;
    }
    .upload button {
      margin-left: 10px;
      padding: 10px 20px;
      border: none;
      background: #a4cbb6;
      color: #fff;
      font-weight: 500;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    .upload button:hover {
      background: #5f9277;
    }
    .prediction {
      text-align: center;
      padding: 20px;
    }
    .prediction h2 {
      color: #4f6158;
      font-size: 1.8rem;
      margin-bottom: 10px;
    }
    .prediction img {
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .confidence-container {
      width: 80%; margin: 10px auto 20px;
      text-align: center;
      font-size: 0.9rem;
      color: #6e6a72;
    }
    .confidence-bar {
      width: 100%; height: 10px;
      background: #e8f0ec;
      border-radius: 5px;
      overflow: hidden;
    }
    .confidence-fill {
      height: 100%;
      background: #8cbfa9;
    }
    .chat-window {
      flex: 1;
      background: #f2f6f2;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
    }
    .message {
      max-width: 75%;
      margin-bottom: 15px;
      padding: 12px 16px;
      border-radius: 18px;
      line-height: 1.4;
      word-break: break-word;
    }
    .user {
      align-self: flex-end;
      background: #93b29f;
      color: #ffffff;
      border-bottom-right-radius: 2px;
    }
    .bot {
      align-self: flex-start;
      background: #d9e9e2;
      color: #374a42;
      border-bottom-left-radius: 2px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .message-content {
      white-space: pre-wrap;
      margin: 0;
    }
    .message-content p {
      margin: 0.2em 0;
    }
    .new-input {
      display: flex;
      padding: 20px;
      border-top: 1px solid #ccd9ce;
      background: #ffffff;
    }
    .new-input input[type=text] {
      flex: 1;
      padding: 12px;
      border: 1px solid #b9cfc2;
      border-radius: 8px;
      font-size: 1rem;
    }
    .new-input button {
      margin-left: 10px;
      padding: 12px 24px;
      border: none;
      background: #a4cbb6;
      color: #ffffff;
      font-weight: 500;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    .new-input button:hover {
      background: #5f9277;
    }
  </style>
</head><body>
  <div class="container">
    <header>Your AI Skincare Companion</header>

    <div class="upload">
      <div>Upload your selfie to begin:</div>
      <br>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br>
        <br>
        <button type="submit">Upload Selfie</button>
      </form>
    </div>

    {% if pred %}
      <div class="prediction">
        <h2>{{ pred }}‑prone skin</h2>
        {% if img %}
          <img src="data:image/png;base64,{{ img }}" width="180">
        {% endif %}
      </div>
      <div class="confidence-container">
        <div class="confidence-bar">
          <div class="confidence-fill" style="width: {{ confidence }}%;"></div>
        </div>
        <div>{{ confidence }}% confidence</div>
      </div>

      <div class="chat-window" id="chat-window">
        {% for msg in history %}
          <div class="message {{ msg.sender }}">
            <div class="message-content">{{ msg.text|safe }}</div>
          </div>
        {% endfor %}
      </div>

      <div class="new-input">
        <form method="post" style="display:flex; width:100%;">
          <input type="text" name="message" placeholder="Ask me anything about your skincare…" autofocus required>
          <button type="submit">Send</button>
        </form>
      </div>
    {% endif %}
  </div>
  <script>
    const cw = document.getElementById('chat-window');
    if (cw) cw.scrollTop = cw.scrollHeight;
  </script>
</body>
</html>
'''

# Route definitions
@app.route('/', methods=['GET','POST'])
def index():
    """
        Handles both image uploads (POST with file) and chat messages (POST with text),
        and serves the main page on GET.
    """
    session.permanent = True # Keep session until browser closed
    if 'history' not in session:
        # Initialize chat history
        session['history'] = []

    # — Image upload & classify —
    if 'file' in request.files and request.files['file'].filename:
        raw = request.files['file'].read()
        pred, confidence = predict_skin_type(raw)

        # Create a thumbnail for display
        thumb = (Image.open(io.BytesIO(raw))
                     .convert('RGB')
                     .resize((180,180)))
        buf = io.BytesIO(); thumb.save(buf, format='PNG')

        # Reset session state
        session['pred']       = pred
        session['confidence'] = confidence
        session['history']    = [{
          'sender':'bot',
          'text': markdown.markdown(f"Hi there! Your skin type is **{pred}**. How can I help?")
        }]
        session.modified = True

        # Render page showing prediction and empty chat
        return render_template_string(HTML,
                                     pred=pred,
                                     img=base64.b64encode(buf.getvalue()).decode(),
                                     confidence=confidence,
                                     history=session['history'])

    # — Chat message handling —
    if 'message' in request.form and session.get('pred'):
        user_msg = request.form['message'].strip()
        html_user = markdown.markdown(user_msg)
        session['history'].append({'sender':'user','text':html_user})

        # Build prompt including skin type and context
        prompt = (
          f"User skin type: {session['pred']}.\n"
          f"{FEW_SHOT_CONTEXT}\n"
          f"{user_msg}"
        )
        reply = chat_with_cot(prompt, max_tokens=200, n=1)
        html_reply = markdown.markdown(reply.strip())
        session['history'].append({'sender':'bot','text':html_reply})

        session.modified = True
        # Re-render the page with updated chat history
        return render_template_string(HTML,
                                     pred=session['pred'],
                                     img=None,
                                     confidence=session['confidence'],
                                     history=session['history'])

    # — Default GET request —
    return render_template_string(HTML,
                                 pred=None,
                                 img=None,
                                 confidence=None,
                                 history=[])

# Entrypoint
if __name__ == "__main__":
    # Run Flask in debug mode on port 8000
    app.run(debug=True, host="0.0.0.0", port=8000)
