# PDF Document Q&A with Gemma Model 📄🤖

## Overview
This Streamlit application allows users to upload multiple PDF documents and ask questions about their content using advanced retrieval-based question-answering technology. Leveraging Groq's Llama3 model and Google's embeddings, the app provides intelligent, context-aware responses.

## 🌟 Features
- Multi-PDF document upload
- Intelligent document embedding
- Context-based question answering
- Retrieval of most relevant document sections
- Response time tracking
- Document similarity search

## 🛠 Prerequisites
- Python 3.8+
- Streamlit
- API Keys:
  - Groq API Key
  - Google Generative AI API Key

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Pdfusion.git
cd Pdfusion
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 🔧 Configuration
- Modify `model_name` in the LLM initialization to use different Groq models
- Adjust text splitter chunk size and overlap as needed
- Customize the prompt template for specific use cases

## 📦 Project Dependencies
- Streamlit
- LangChain
- Groq
- Google Generative AI
- PyPDF2
- FAISS
- python-dotenv

## 🛠 Customization
- Change embedding model
- Modify retrieval chain parameters
- Adjust response formatting

## 💡 How It Works
1. Upload PDF documents
2. Click "Load Documents"
3. Embeddings are created using Google's embedding model
4. FAISS vector store is generated
5. Enter your question
6. Receive context-based answers

## 🔍 Debugging
- Ensure API keys are valid
- Check PDF file compatibility
- Verify dependencies are correctly installed

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License
Distributed under the MIT License.



## 📚 Additional Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Groq API Documentation](https://console.groq.com/docs)
```
