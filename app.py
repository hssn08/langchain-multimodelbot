import os
import streamlit as st
from typing import List, Dict, Any, Optional

# LangChain imports with updated import paths
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and LLM models
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Safe rerun function to handle different Streamlit versions
def safe_rerun():
    """
    Safely rerun the app regardless of streamlit version.
    Works with both older versions using experimental_rerun and newer versions using rerun.
    """
    try:
        # Try the newer method first
        st.rerun()
    except AttributeError:
        # Fall back to the older method
        try:
            st.experimental_rerun()
        except AttributeError:
            # If both fail, just show a message
            st.warning("Couldn't automatically refresh. Please refresh the page manually.")

# Set up API keys from Streamlit secrets
if 'OPENAI_API_KEY' in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    # For local development, you can use .streamlit/secrets.toml
    pass

if 'ANTHROPIC_API_KEY' in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
else:
    # For local development, you can use .streamlit/secrets.toml
    pass

if 'GOOGLE_API_KEY' in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # For local development, you can use .streamlit/secrets.toml
    pass

def load_pdf_with_advanced_recovery(pdf_file):
    """
    Attempts to load a PDF using multiple methods with robust error handling.
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        List of Document objects or None if all methods fail
    """
    import tempfile
    import os
    import traceback
    from langchain_core.documents import Document
    
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    documents = None
    error_messages = []
    
    try:
        # Method 1: Try PyPDFLoader (standard approach)
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if documents:
                print(f"Successfully loaded PDF with PyPDFLoader: {len(documents)} pages")
                os.unlink(pdf_path)
                return documents
        except Exception as e:
            error_message = f"PyPDFLoader failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
        
        # Method 2: Try PyPDF2 with strict=False
        try:
            import PyPDF2
            documents = []
            with open(pdf_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file, strict=False)
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():  # Only add non-empty pages
                            documents.append(Document(
                                page_content=text,
                                metadata={"page": i, "source": pdf_path}
                            ))
                    if documents:
                        print(f"Successfully loaded PDF with PyPDF2: {len(documents)} pages")
                        os.unlink(pdf_path)
                        return documents
                except Exception as pdf_error:
                    error_message = f"PyPDF2 failed: {str(pdf_error)}"
                    error_messages.append(error_message)
                    print(error_message)
        except Exception as e:
            error_message = f"PyPDF2 method failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
        
        # Method 3: Try pdfplumber
        try:
            import pdfplumber
            documents = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():  # Only add non-empty pages
                        documents.append(Document(
                            page_content=text,
                            metadata={"page": i, "source": pdf_path}
                        ))
                if documents:
                    print(f"Successfully loaded PDF with pdfplumber: {len(documents)} pages")
                    os.unlink(pdf_path)
                    return documents
        except Exception as e:
            error_message = f"pdfplumber failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
            
        # Method 4: Try pdf2image + pytesseract (OCR approach)
        try:
            import pdf2image
            import pytesseract
            from PIL import Image
            
            documents = []
            images = pdf2image.convert_from_path(pdf_path)
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                if text and text.strip():  # Only add non-empty pages
                    documents.append(Document(
                        page_content=text,
                        metadata={"page": i, "source": pdf_path, "extraction_method": "OCR"}
                    ))
            if documents:
                print(f"Successfully loaded PDF with OCR: {len(documents)} pages")
                os.unlink(pdf_path)
                return documents
        except Exception as e:
            error_message = f"OCR method failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
            
        # Method 5: Try binary analysis (last resort)
        try:
            with open(pdf_path, 'rb') as file:
                binary_data = file.read()
                
                # Check if file might be password protected
                if b'/Encrypt' in binary_data:
                    error_message = "PDF appears to be password protected"
                    error_messages.append(error_message)
                    print(error_message)
                
                # Check file signature/magic bytes
                if not binary_data.startswith(b'%PDF'):
                    # Not a standard PDF file, might be corrupted or different format
                    error_message = f"File doesn't have a standard PDF header. First bytes: {binary_data[:20]}"
                    error_messages.append(error_message)
                    print(error_message)
        except Exception as e:
            error_message = f"Binary analysis failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
            
    except Exception as e:
        error_message = f"Overall PDF processing failed: {str(e)}"
        error_messages.append(error_message)
        print(error_message)
        print(traceback.format_exc())
    
    # Clean up temp file if we get here (all methods failed)
    try:
        os.unlink(pdf_path)
    except:
        pass
        
    # Compile detailed error report
    detailed_error = "\n".join(error_messages)
    print(f"All PDF extraction methods failed. Details:\n{detailed_error}")
    
    return None

class PDFChatbot:
    def __init__(self, system_prompt: str = None, system_prompt_file: str = None, model_name: str = "openai"):
        """
        Initialize the chatbot with a system prompt and model name.
        
        Args:
            system_prompt: The system prompt to guide the model's behavior (string)
            system_prompt_file: Path to a text file containing the system prompt
            model_name: 'openai', 'anthropic', or 'gemini' to select the LLM provider
        """
        if system_prompt_file:
            try:
                with open(system_prompt_file, 'r', encoding='utf-8') as file:
                    self.system_prompt = file.read()
            except Exception as e:
                st.error(f"Error loading system prompt file: {e}")
                self.system_prompt = "You are a helpful assistant that answers questions based on the provided PDF document."
        else:
            self.system_prompt = system_prompt or "You are a helpful assistant that answers questions based on the provided PDF document."
            
        self.chat_history = ChatMessageHistory()
        
        # Instead of deprecated ConversationBufferMemory, we'll manage memory manually
        self.messages = []
        
        # Initialize with the specified model
        self.set_model(model_name)
        self.embeddings = OpenAIEmbeddings()
        self.retriever = None
        
    def set_model(self, model_name: str) -> None:
        """
        Set the LLM model to use.
        
        Args:
            model_name: 'openai', 'anthropic', or 'gemini'
        """
        if model_name.lower() == "openai":
            self.llm = ChatOpenAI(model="gpt-4o")
            self.model_name = "openai"
        elif model_name.lower() == "anthropic":
            self.llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")
            self.model_name = "anthropic"
        elif model_name.lower() == "gemini":
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.model_name = "gemini"
        else:
            raise ValueError("Model must be 'openai', 'anthropic', or 'gemini'")
    
    def load_pdf(self, pdf_file) -> None:
        """
        Load and process a PDF document from an uploaded file using advanced recovery methods.
        
        Args:
            pdf_file: Uploaded PDF file object
        """
        with st.spinner("Processing PDF... This may take a moment."):
            try:
                # Try using advanced PDF recovery
                documents = load_pdf_with_advanced_recovery(pdf_file)
                
                if not documents or len(documents) == 0:
                    # Try one more approach with pikepdf
                    try:
                        import tempfile
                        import pikepdf
                        import PyPDF2
                        from langchain_core.documents import Document
                        
                        # Create a temporary file for the original PDF
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(pdf_file.getvalue())
                            original_path = tmp_file.name
                        
                        # Create a temporary file for the repaired PDF
                        repaired_path = original_path + ".repaired.pdf"
                        
                        # Try to repair the PDF with pikepdf
                        with pikepdf.open(original_path, allow_overwriting_input=True) as pdf:
                            pdf.save(repaired_path)
                        
                        # Try to extract text from the repaired PDF
                        documents = []
                        with open(repaired_path, 'rb') as file:
                            try:
                                pdf_reader = PyPDF2.PdfReader(file, strict=False)
                                for i, page in enumerate(pdf_reader.pages):
                                    text = page.extract_text()
                                    if text and text.strip():
                                        documents.append(Document(
                                            page_content=text,
                                            metadata={"page": i, "source": repaired_path}
                                        ))
                            except Exception as e:
                                st.error(f"Error extracting text from repaired PDF: {e}")
                        
                        # Clean up temporary files
                        try:
                            os.unlink(original_path)
                            os.unlink(repaired_path)
                        except:
                            pass
                            
                        if not documents or len(documents) == 0:
                            st.error("Could not extract text from the PDF after repair attempts.")
                            return
                            
                    except ImportError:
                        st.error("pikepdf is not installed. Install with: pip install pikepdf")
                        return
                    except Exception as e:
                        st.error(f"Error during PDF repair: {e}")
                        return
                
                # Split the document into chunks
                try:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    if not chunks:
                        st.error("Could not create text chunks from the PDF.")
                        return
                    
                    # Create a vector store
                    vectorstore = FAISS.from_documents(chunks, self.embeddings)
                    
                    # Set up the retriever
                    self.retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    
                    st.success(f"PDF processed and indexed with {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error during text splitting or indexing: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    def create_prompt(self) -> ChatPromptTemplate:
        """
        Create the chat prompt template with system message and context.
        
        Returns:
            ChatPromptTemplate: The configured prompt template
        """
        template = """
        {system_prompt}
        
        Context information from the document:
        {context}
        
        Chat History:
        {chat_history}
        
        Human: {question}
        AI Assistant:
        """
        
        return ChatPromptTemplate.from_template(template)

    def format_chat_history(self) -> str:
        """
        Format the chat history for including in the prompt.
        
        Returns:
            str: Formatted chat history
        """
        formatted_history = ""
        messages = self.chat_history.messages
        
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"AI: {message.content}\n"
        
        return formatted_history
    
    def generate_context(self, question: str) -> str:
        """
        Generate context from the document based on the question.
        
        Args:
            question: The user's question
            
        Returns:
            str: Relevant context from the document
        """
        if not self.retriever:
            return "No document has been loaded yet."
        
        # Using .invoke() instead of deprecated get_relevant_documents()
        relevant_docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        return context
    
    def chat(self, question: str) -> str:
        """
        Process a user question and generate a response.
        
        Args:
            question: The user's question
            
        Returns:
            str: The AI's response
        """
        # Add the user message to history
        self.chat_history.add_user_message(question)
        
        # If no document is loaded, use a simplified chain without document context
        if not self.retriever:
            # Create a simple prompt for general conversation
            simple_template = """
            {system_prompt}
            
            Chat History:
            {chat_history}
            
            Human: {question}
            AI Assistant:
            """
            
            simple_prompt = ChatPromptTemplate.from_template(simple_template)
            chat_history = self.format_chat_history()
            
            # Create a simplified chain
            simple_chain = (
                {
                    "system_prompt": lambda _: self.system_prompt,
                    "chat_history": lambda _: chat_history,
                    "question": lambda x: x
                }
                | simple_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Run the chain
            with st.spinner("Thinking..."):
                response = simple_chain.invoke(question)
            
            # Add AI response to history
            self.chat_history.add_ai_message(response)
            
            return response
        
        # Generate context from relevant document parts
        context = self.generate_context(question)
        
        # Format chat history
        chat_history = self.format_chat_history()
        
        # Create the prompt
        prompt = self.create_prompt()
        
        # Create the chain
        chain = (
            {
                "system_prompt": lambda _: self.system_prompt,
                "context": lambda _: context,
                "chat_history": lambda _: chat_history,
                "question": lambda x: x
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Run the chain
        with st.spinner("Thinking..."):
            response = chain.invoke(question)
        
        # Add AI response to history
        self.chat_history.add_ai_message(response)
        
        return response

    def clear_history(self):
        """Clear the chat history"""
        self.chat_history = ChatMessageHistory()
        self.messages = []


def main():
    st.set_page_config(page_title="PDF AI Chatbot", page_icon="📚", layout="wide")
    
    # Improve UI with custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .model-selector {
        margin-bottom: 10px;
    }
    .chat-container {
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and description
    st.title("📚 PDF AI Chatbot")
    st.markdown("Chat with your PDF documents using AI. Powered by LangChain, OpenAI, Anthropic, and Google Gemini 2.0 Flash.")
    
    # Initialize session state for storing the chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.messages = []
    
    # Check if API keys are available
    models_available = []
    
    if 'OPENAI_API_KEY' in os.environ or 'OPENAI_API_KEY' in st.secrets:
        models_available.append("OpenAI (GPT-4o)")
    
    if 'ANTHROPIC_API_KEY' in os.environ or 'ANTHROPIC_API_KEY' in st.secrets:
        models_available.append("Anthropic (Claude 3.7 Sonnet)")
    
    if 'GOOGLE_API_KEY' in os.environ or 'GOOGLE_API_KEY' in st.secrets:
        models_available.append("Google (Gemini 2.0 Flash)")
    
    if not models_available:
        st.error("⚠️ No API keys are set. Please set at least one API key in your Streamlit secrets.")
        st.stop()
        
    st.sidebar.write(f"Available models: {', '.join(models_available)}")
    
    # Create sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection based on available API keys
        models_available = []
        
        if 'OPENAI_API_KEY' in os.environ or 'OPENAI_API_KEY' in st.secrets:
            models_available.append("OpenAI (GPT-4o)")
        
        if 'ANTHROPIC_API_KEY' in os.environ or 'ANTHROPIC_API_KEY' in st.secrets:
            models_available.append("Anthropic (Claude 3.7 Sonnet)")
        
        if 'GOOGLE_API_KEY' in os.environ or 'GOOGLE_API_KEY' in st.secrets:
            models_available.append("Google (Gemini 2.0 Flash)")
        
        model_name = st.selectbox(
            "Select AI Model",
            options=models_available,
            index=0
        )
        
        # Map friendly names to model identifiers
        model_mapping = {
            "OpenAI (GPT-4o)": "openai",
            "Anthropic (Claude 3.7 Sonnet)": "anthropic",
            "Google (Gemini 2.0 Flash)": "gemini"
        }
        
        # System prompt input (text area or file upload)
        st.subheader("System Prompt")
        prompt_method = st.radio(
            "Choose input method:",
            options=["Text Input", "Upload File"]
        )
        
        system_prompt = None
        system_prompt_file = None
        
        if prompt_method == "Text Input":
            system_prompt = st.text_area(
                "Enter system prompt:",
                value="You are a helpful assistant that answers questions based on the provided PDF document. Always be concise, accurate, and helpful. If you don't know the answer or the information is not in the document, say so.",
                height=150
            )
        else:
            uploaded_prompt = st.file_uploader("Upload system prompt text file", type=["txt"])
            if uploaded_prompt:
                system_prompt = uploaded_prompt.getvalue().decode("utf-8")
        
        # PDF upload
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        # Initialize or reinitialize the chatbot
        if st.button("Initialize Chatbot"):
            with st.spinner("Initializing..."):
                st.session_state.chatbot = PDFChatbot(
                    system_prompt=system_prompt,
                    model_name=model_mapping[model_name]
                )
                st.session_state.messages = []
                st.success(f"Chatbot initialized with {model_name} model")
                st.info("You can now start chatting! Upload a PDF anytime if you want to discuss a document.")
        
        # Load PDF if uploaded in the sidebar
        if uploaded_file and st.session_state.chatbot:
            if st.button("Process PDF", key="sidebar_process_pdf"):
                st.session_state.chatbot.load_pdf(uploaded_file)
        
        # Clear chat history
        if st.session_state.chatbot and st.button("Clear Chat History"):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.success("Chat history cleared")
        
        # Add help text about model switching
        st.info("You can also switch models directly in the chat interface after initializing the chatbot.")
    
    # Main chat area
    st.header("💬 Chat")
    
    # Add model selector in main area for easy switching
    if st.session_state.chatbot:
        # Create columns for the controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Model selection dropdown in main area
            models_available = []
            if 'OPENAI_API_KEY' in os.environ or 'OPENAI_API_KEY' in st.secrets:
                models_available.append("OpenAI (GPT-4o)")
            if 'ANTHROPIC_API_KEY' in os.environ or 'ANTHROPIC_API_KEY' in st.secrets:
                models_available.append("Anthropic (Claude 3.7 Sonnet)")
            if 'GOOGLE_API_KEY' in os.environ or 'GOOGLE_API_KEY' in st.secrets:
                models_available.append("Google (Gemini 2.0 Flash)")
            
            # Get current model display name
            current_model_name = next((name for name, id in {
                "OpenAI (GPT-4o)": "openai",
                "Anthropic (Claude 3.7 Sonnet)": "anthropic",
                "Google (Gemini 2.0 Flash)": "gemini"
            }.items() if id == st.session_state.chatbot.model_name), models_available[0])
            
            new_model = st.selectbox(
                "Select AI model:",
                options=models_available,
                index=models_available.index(current_model_name) if current_model_name in models_available else 0,
                key="main_model_selector"
            )
        
        with col2:
            # Button to switch models
            if st.button("Switch Model", key="main_switch_model"):
                model_mapping = {
                    "OpenAI (GPT-4o)": "openai",
                    "Anthropic (Claude 3.7 Sonnet)": "anthropic",
                    "Google (Gemini 2.0 Flash)": "gemini"
                }
                new_model_id = model_mapping[new_model]
                if new_model_id != st.session_state.chatbot.model_name:
                    st.session_state.chatbot.set_model(new_model_id)
                    st.success(f"Switched to {new_model}")
                    safe_rerun()  # Use safe rerun function
        
        with col3:
            # Clear chat button
            if st.button("Clear Chat", key="main_clear_chat"):
                st.session_state.chatbot.clear_history()
                st.session_state.messages = []
                st.success("Chat history cleared")
                safe_rerun()  # Use safe rerun function
    
    # PDF upload capability in the main area
    if st.session_state.chatbot:
        with st.expander("📄 Upload PDF Document", expanded=False):
            main_uploaded_file = st.file_uploader("Choose a PDF file to chat with", type=["pdf"], key="main_pdf_uploader")
            if main_uploaded_file:
                if st.button("Process PDF", key="main_process_pdf"):
                    st.session_state.chatbot.load_pdf(main_uploaded_file)
    
    # Display warning if chatbot is not initialized
    if not st.session_state.chatbot:
        st.warning("Please initialize the chatbot from the sidebar first.")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about your document...")
        
        if user_input:
            # Add user message to UI
            with st.chat_message("user"):
                st.write(user_input)
            
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get response from chatbot
            response = st.session_state.chatbot.chat(user_input)
            
            # Add assistant response to UI
            with st.chat_message("assistant"):
                st.write(response)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()