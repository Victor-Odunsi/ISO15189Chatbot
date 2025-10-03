# ISO 15189:2022 RAG Chatbot – Laboratory Quality Management Assistant  

## 📌 Overview  
As a **medical laboratory scientist**, I recognize how critical **Quality Management Systems (QMS)** are for ensuring accurate results, patient safety, and accreditation. However, navigating the **ISO 15189:2022** standard can be challenging for professionals seeking practical guidance.  

This project uses **Retrieval-Augmented Generation (RAG)** to build a chatbot that provides **clause-specific, contextual answers** from the ISO 15189:2022 document. Users can ask questions such as:  
> *"What does ISO 15189 say about equipment calibration?"*  

The chatbot retrieves the relevant clause, generates an easy-to-understand explanation, and provides reference to the official text.  

👉 **[Live Demo](http://13.62.69.28:8501/)**  

---

## ⚡ Features  
- Clause-level retrieval from ISO 15189:2022  
- Contextual explanations powered by LLMs  
- Streamlit-based interactive chatbot interface  
- Support for practical compliance and implementation guidance  

---

## 🛠️ Tech Stack & Skills  
- **LangChain** – RAG pipeline orchestration  
- **Chroma** – vector database for semantic retrieval  
- **OpenAI LLMs** – contextual Q&A generation  
- **Python** – core implementation language  
- **Streamlit** – web-based user interface  
- **PDF Processing (PyMuPDF, pdfplumber)** – document parsing and chunking  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.11.9  
- GROQ API key
- MISTRALAI API Key

### Installation  
```bash
# Clone repository
git clone https://github.com/your-username/iso15189-rag-chatbot.git
cd iso15189-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
