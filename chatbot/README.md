# Insurance Documents QA RAG Chatbot  
Your Intelligent Chat Assistant for Insurance Document Queries  

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)

---

## ✨ About the Project  
The Insurance Documents QA RAG Chatbot is an advanced conversational assistant designed to streamline complex queries on insurance documents while also providing general information by conducting internet searches. Utilizing Retrieval-Augmented Generation (RAG) pipelines, the chatbot intelligently determines whether to retrieve information from a document or the web, ensuring accurate and contextually relevant responses.

---

## 🔍 Key Features  
- ✨ **Dual-Functionality Query Resolution**: Handles document-specific and general queries with smart tool selection.  
- ☆ **Efficient Retrieval**: Employs **LlamaIndex** for accurate and efficient insurance data retrieval.  
- ✪ **Agentic Generation**: Uses **LangChain** to intelligently generate responses or search the web for answers.  
- ✨ **Scalable Embedding Storage**: Leverages **ChromaDB** to store and query embeddings with speed and accuracy.  
- 🔑 **Advanced Caching Mechanism**: Minimizes redundant computations with:  
  - Embedding-level cache for reprocessing prevention.  
  - Query/answer cache for repeated queries.  
- 🔄 **Dynamic Chunking**: Splits documents for optimal information retrieval.  
- 🕒 **Real-Time Adaptability**: Decides between data retrieval and internet search based on query type.

---

## 🛠️ Tech Stack  
- **Language**: Python  
- **Frameworks/Libraries**: LlamaIndex, LangChain, ChromaDB
- **APIs/Models**:  
  - OpenAI's Embedding Model for vector creation 
  - OpenAI for high-quality generative responses
  - LlamaIndex for document ingestion and querying
  - LangChain for agent orchestration.
  - ChromaDB for efficient data storage.

---

## 🧪 Example Use Cases   
- "What are the exclusions in my health insurance policy?"  
- "How do I file an insurance claim?"  
- "What is Drug Abuse related death?" (via web search) 

---

## 📸 Sample Output  
### 1. Insurance Query Response  
![Insurance Query Response](Sample%20Code%20Output%20Screenshots/Code%20Output%201.png)  

### 2. General Query Response with Web Search  
![General Query Response](Sample%20Code%20Output%20Screenshots/Code%20Output%202.png)  

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation
1. Clone the repo:
git clone https://github.com/SandeepGitGuy/Insurance_Documents_QA_Chatbot_RAG_LlamaIndex_LangChain.git

2. Navigate to the project directory:
cd Insurance_Documents_QA_Chatbot_RAG_LlamaIndex_LangChain

3. Install the required dependencies:
pip install -r requirements.txt

- Please note: OpenAI API keys are required for the project to function.

4. Run the main file from Jupyter environment:
"Insurance_Docs_QA_Chatbot_RAG_Llamaindex_LangChain.ipynb"

---

## 🚂 Challenges Faced and Fixes  
- **Dynamic Tool Selection**: Implemented robust logic to decide between document retrieval and internet search.  
- **Caching Efficiency**: Added efficient multi-layer caching to optimize embedding and query-answer processes.  
- **Cross-Model Compatibility**: Fine-tuned embeddings and outputs to work seamlessly with ChromaDB and OpenAI APIs.  
- **Error Recovery**: Implemented memory-driven recovery mechanisms for enhanced reliability.  

---

## 🌐 Future Scope  
- Support for multilingual documents and queries.  
- Integration with additional file formats like Word and Excel.  
- Incorporation of more generative AI models.  

---

## 🔗 Documentation  
No documentation will be made available for this project since this project only uses technologies that already have their own documentation. Please refer to the following links for more information:
- [LlamaIndex](https://llamaindex.ai/)  
- [LangChain](https://langchain.com/)  
- [ChromaDB](https://docs.trychroma.com/)

---

## 🛡️ Conclusion  
The Insurance Documents QA RAG Chatbot leverages state-of-the-art tools like LangChain and LlamaIndex to create a seamless query resolution experience. With its dual ability to handle document-based and general queries, it simplifies complex interactions while maintaining high performance and scalability. By integrating advanced caching and retrieval mechanisms, this chatbot offers an efficient, user-friendly solution for navigating insurance-related and general inquiries. Its robust foundation sets the stage for future innovations in AI-powered assistants.

---

## 🛡️ License  
Distributed under the MIT License. See `LICENSE` for more details.  

---

## 💬 Contact  
For questions, feedback, or collaboration opportunities:  
- **Email**: sandy974278@gmail.com  
- **GitHub**: [SandeepGitGuy](https://github.com/SandeepGitGuy)  
- **LinkedIn**: [Sandeep Gowda](https://www.linkedin.com/in/sandeepgowda24a319192)  

---