# 🧠 MemoryAIChat – Your AI-Powered Visual Memory Assistant

Ever wish you could revisit a moment from your life with the clarity of a vivid dream — or ask an AI about a memory the way you'd ask a close friend?

**MemoryAIChat** is an experimental AI chatbot that acts as your personal **time capsule** and **memory companion**, combining advanced multimodal AI with vector similarity search to help you **log, retrieve, and explore your memories** in a meaningful way.

Powered by cutting-edge models like **Meta’s LLaMA 3.2 11B Vision Instruct**, **OpenAI’s GPT-4**, and **Azure Cognitive Services**, this assistant can take voice, image, and text inputs and **recall relevant personal moments** you've logged over time.

---

## ✨ Features

* 🔹 **Visual Memory Logging**
  Upload a photo, speak your thoughts, or type out an experience — and MemoryAIChat will store it as a retrievable memory.

* 🔹 **AI-Powered Memory Recall**
  Ask the assistant questions like *"What did I do in Tokyo in April?"* and it will **search your memory store** using **ChromaDB** and **vector similarity** to bring up relevant experiences and visuals.

* 🔹 **Multimodal Interaction**
  Speak your thoughts using integrated **Azure Speech SDK**, or engage the AI through chat. It responds with empathy, context, and vision-aware insights.

* 🔹 **Memory as Time Capsule**
  Treat the assistant like a digital diary you can query — helping you reflect, reminisce, and make sense of your past.

---

## 💠 Tech Stack

* 🧠 **LLaMA 3.2 11B Vision Instruct** – Multimodal reasoning
* 🤖 **OpenAI GPT** – Language understanding and generation
* 🛞 **Azure Speech SDK** – Voice-to-text memory input
* 🔍 **ChromaDB** – Vector-based memory retrieval
* 🎯 **Streamlit** – Interactive web interface

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/taliandre49/MemoryAIChat.git
cd MemoryAIChat
```

### 2. Set up your `.env` file

Create a `.env` file at the root level and add the necessary keys:

```env
OPENAI_API_KEY=your_openai_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION = your_region
LLAMA_API_URL=your_llama_endpoint
...
```

> Each API provider (OpenAI, Azure, Meta) may require registration to obtain credentials. Ensure your keys have sufficient usage permissions.

### 3. Install dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run welcome.py
```

---

## 📸 Example Use Cases

* “Log a memory from today with this photo and my voice.”
* “Remind me of times I mentioned my dog in voice notes.”
* “Did I ever visit the beach with Mingyi?”
* “What memories are tagged with ‘joyful’ and ‘New York’?”

---

## 🤝 Contributors

This project was proudly built with contribution of the following individuals:

* **Shania Cabrera**
* **Mingyi Shao**
* **Zaeda Amrin**
* **Jai Kishore Kumar Chandnani**

---

> 📌 *MemoryAIChat is an open playground for exploring the future of human-AI memory interaction. This project is experimental and not intended for production use without further refinement.*

