import json
import random
import nltk
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents
with open("intents.json", encoding="utf-8") as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()

# Global mute flag
muted = False

# Preprocess training data
X = []
y = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        X.append(" ".join(lemmas))
        y.append(intent["tag"])

# Train model
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vector, y)

# Predict response
def get_response(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    input_vector = vectorizer.transform([" ".join(lemmas)])
    predicted_tag = model.predict(input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# ✅ Speak function: always creates a fresh engine
def speak(text):
    if not muted:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("Speech error:", e)

# GUI send function
def send():
    user_input = entry.get().strip()
    if not user_input:
        return
    chat_window.insert(tk.END, f"You: {user_input}\n")

    response = get_response(user_input)
    chat_window.insert(tk.END, f"Bot: {response}\n")
    chat_window.see(tk.END)

    speak(response)
    entry.delete(0, tk.END)

# Mute toggle
def toggle_mute():
    global muted
    muted = not muted
    mute_button.config(text="Unmute" if muted else "Mute")

# GUI Setup
root = tk.Tk()
root.title("Voice Chatbot")
root.configure(bg="lightyellow")

chat_window = scrolledtext.ScrolledText(root, width=80, height=20, font=("Arial", 14), bg="lightyellow")
chat_window.pack(padx=10, pady=10)

frame = tk.Frame(root, bg="lightyellow")
frame.pack(pady=5)

entry = tk.Entry(frame, width=60, font=("Arial", 14))
entry.pack(side=tk.LEFT, padx=(10, 5))

send_button = tk.Button(frame, text="Send", command=send, bg="lightblue", font=("Arial", 12))
send_button.pack(side=tk.LEFT, padx=5)

mute_button = tk.Button(frame, text="Mute", command=toggle_mute, bg="lightgray", font=("Arial", 12))
mute_button.pack(side=tk.LEFT, padx=5)

root.mainloop()
