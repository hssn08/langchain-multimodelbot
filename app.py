import os
import streamlit as st
from typing import List, Dict, Any, Optional

# LangChain imports with updated import paths
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# Embeddings and LLM models
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
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

def extract_text_from_pdf(pdf_file):
    """
    Attempts to extract all text from a PDF using multiple methods with robust error handling.
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        String containing all text from the PDF or None if all methods fail
    """
    import tempfile
    import os
    import traceback
    
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    full_text = ""
    error_messages = []
    
    try:
        # Method 1: Try PyPDFLoader
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if documents:
                full_text = "\n\n".join([doc.page_content for doc in documents])
                print(f"Successfully extracted text with PyPDFLoader: {len(documents)} pages")
                os.unlink(pdf_path)
                return full_text
        except Exception as e:
            error_message = f"PyPDFLoader failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
        
        # Method 2: Try PyPDF2 with strict=False
        try:
            import PyPDF2
            extracted_text = []
            with open(pdf_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file, strict=False)
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():  # Only add non-empty pages
                            extracted_text.append(text)
                    if extracted_text:
                        full_text = "\n\n".join(extracted_text)
                        print(f"Successfully extracted text with PyPDF2: {len(extracted_text)} pages")
                        os.unlink(pdf_path)
                        return full_text
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
            extracted_text = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():  # Only add non-empty pages
                        extracted_text.append(text)
                if extracted_text:
                    full_text = "\n\n".join(extracted_text)
                    print(f"Successfully extracted text with pdfplumber: {len(extracted_text)} pages")
                    os.unlink(pdf_path)
                    return full_text
        except Exception as e:
            error_message = f"pdfplumber failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
            
        # Method 4: Try pdf2image + pytesseract (OCR approach)
        try:
            import pdf2image
            import pytesseract
            from PIL import Image
            
            extracted_text = []
            images = pdf2image.convert_from_path(pdf_path)
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                if text and text.strip():  # Only add non-empty pages
                    extracted_text.append(text)
            if extracted_text:
                full_text = "\n\n".join(extracted_text)
                print(f"Successfully extracted text with OCR: {len(extracted_text)} pages")
                os.unlink(pdf_path)
                return full_text
        except Exception as e:
            error_message = f"OCR method failed: {str(e)}"
            error_messages.append(error_message)
            print(error_message)
            
        # Method 5: Try pikepdf (repair approach)
        try:
            import pikepdf
            import PyPDF2
            
            # Create a temporary file for the repaired PDF
            repaired_path = pdf_path + ".repaired.pdf"
            
            # Try to repair the PDF with pikepdf
            with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
                pdf.save(repaired_path)
            
            # Try to extract text from the repaired PDF
            extracted_text = []
            with open(repaired_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file, strict=False)
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            extracted_text.append(text)
                    if extracted_text:
                        full_text = "\n\n".join(extracted_text)
                        print(f"Successfully extracted text from repaired PDF: {len(extracted_text)} pages")
                        
                        # Clean up temporary files
                        try:
                            os.unlink(pdf_path)
                            os.unlink(repaired_path)
                        except:
                            pass
                            
                        return full_text
                except Exception as e:
                    error_message = f"Error extracting text from repaired PDF: {e}"
                    error_messages.append(error_message)
                    print(error_message)
        except Exception as e:
            error_message = f"PDF repair method failed: {str(e)}"
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

class FullContextPDFChatbot:
    def __init__(self, system_prompt: str = None, model_name: str = "openai"):
        """
        Initialize the chatbot with a system prompt and model name.
        
        Args:
            system_prompt: The system prompt to guide the model's behavior (string)
            model_name: 'openai', 'anthropic', or 'gemini' to select the LLM provider
        """
        self.system_prompt = system_prompt or "You are a helpful assistant that answers questions based on the provided PDF document."
        self.chat_history = ChatMessageHistory()
        
        # Initialize with the specified model
        self.set_model(model_name)
        
        # Store PDF content
        self.pdf_contents = {}  # Dictionary to store multiple PDFs: {filename: content}
        self.pdf_loaded = False
        
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
        Load and process a single PDF document from an uploaded file.
        
        Args:
            pdf_file: Uploaded PDF file object
        """
        with st.spinner(f"Extracting text from {pdf_file.name}... This may take a moment."):
            try:
                # Extract full text from PDF
                full_text = extract_text_from_pdf(pdf_file)
                
                if not full_text:
                    st.error(f"Could not extract text from {pdf_file.name} after multiple attempts.")
                    return
                
                # Store the full text with filename as key
                self.pdf_contents[pdf_file.name] = full_text
                self.pdf_loaded = True
                
                # Provide information about the document
                word_count = len(full_text.split())
                page_estimate = max(1, word_count // 500)  # Rough estimate of page count
                
                st.success(f"PDF '{pdf_file.name}' processed successfully! Extracted approximately {word_count} words (~{page_estimate} pages).")
                
                # Update session state to store loaded PDF names
                if 'loaded_pdfs' not in st.session_state:
                    st.session_state.loaded_pdfs = []
                
                if pdf_file.name not in st.session_state.loaded_pdfs:
                    st.session_state.loaded_pdfs.append(pdf_file.name)
                
            except Exception as e:
                st.error(f"Error processing PDF '{pdf_file.name}': {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    def load_multiple_pdfs(self, pdf_files) -> None:
        """
        Load and process multiple PDF documents from uploaded files.
        
        Args:
            pdf_files: List of uploaded PDF file objects
        """
        for pdf_file in pdf_files:
            self.load_pdf(pdf_file)
    
    def remove_pdf(self, filename) -> None:
        """
        Remove a PDF from the loaded documents.
        
        Args:
            filename: Name of the PDF to remove
        """
        if filename in self.pdf_contents:
            del self.pdf_contents[filename]
            
            # Update loaded_pdfs in session state
            if 'loaded_pdfs' in st.session_state and filename in st.session_state.loaded_pdfs:
                st.session_state.loaded_pdfs.remove(filename)
            
            st.success(f"Removed '{filename}' from loaded documents.")
            
            # Update pdf_loaded flag
            self.pdf_loaded = len(self.pdf_contents) > 0
        else:
            st.warning(f"PDF '{filename}' not found in loaded documents.")

    def create_prompt(self) -> ChatPromptTemplate:
        """
        Create the chat prompt template with system message and full document context.
        
        Returns:
            ChatPromptTemplate: The configured prompt template
        """
        template = """
        {system_prompt}
        
        Document Content:
        {document_content}
        
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
    
    def combine_pdf_contents(self) -> str:
        """
        Combine all loaded PDF contents into a single string with document markers.
        
        Returns:
            str: Combined PDF content with document markers
        """
        if not self.pdf_contents:
            return ""
        
        combined_text = ""
        
        # Add each document with a header
        for filename, content in self.pdf_contents.items():
            combined_text += f"\n\n--- DOCUMENT: {filename} ---\n\n"
            combined_text += content
        
        return combined_text
    
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
        if not self.pdf_loaded:
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
        
        # Format chat history
        chat_history = self.format_chat_history()
        
        # Combine all PDF contents with document markers
        combined_pdf_content = self.combine_pdf_contents()
        
        # Create the prompt with full document context
        prompt = self.create_prompt()
        
        # Create the chain
        chain = (
            {
                "system_prompt": lambda _: self.system_prompt,
                "document_content": lambda _: combined_pdf_content,
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


def main():
    st.set_page_config(page_title="Full Context PDF AI Chatbot", page_icon="📚", layout="wide")
    
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
    .pdf-list {
        margin-top: 10px;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and description
    st.title("📚 Full Context PDF AI Chatbot")
    st.markdown("Chat with your PDF documents using AI. Uses full document context without vector databases.")
    
    # Initialize session state for storing the chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.messages = []
        st.session_state.loaded_pdfs = []
    
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
    
    # Create sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection based on available API keys
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
                value="You are a helpful assistant that answers questions based on the provided PDF documents. Always be accurate and reference specific parts of the documents when answering questions. If you don't know the answer or the information is not in the documents, say so. When referencing information, mention which document it came from.",
                height=150
            )
        else:
            uploaded_prompt = st.file_uploader("Upload system prompt text file", type=["txt"])
            if uploaded_prompt:
                system_prompt = uploaded_prompt.getvalue().decode("utf-8")
        
        # Initialize or reinitialize the chatbot
        if st.button("Initialize Chatbot"):
            with st.spinner("Initializing..."):
                st.session_state.chatbot = FullContextPDFChatbot(
                    system_prompt=system_prompt,
                    model_name=model_mapping[model_name]
                )
                st.session_state.messages = []
                st.success(f"Chatbot initialized with {model_name} model")
                st.info("You can now start chatting! Upload PDFs anytime if you want to discuss documents.")
        
        # PDF upload section in sidebar
        if st.session_state.chatbot:
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
            
            if uploaded_files:
                if st.button("Process PDFs", key="sidebar_process_pdfs"):
                    for pdf_file in uploaded_files:
                        st.session_state.chatbot.load_pdf(pdf_file)
        
        # Clear chat history
        if st.session_state.chatbot and st.button("Clear Chat History"):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.success("Chat history cleared")
    
    # Main chat area
    st.header("💬 Chat")
    
    # Add model selector in main area for easy switching
    if st.session_state.chatbot:
        # Create columns for the controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Model selection dropdown in main area
            current_model_name = next((name for name, id in model_mapping.items() 
                                      if id == st.session_state.chatbot.model_name), 
                                     models_available[0])
            
            new_model = st.selectbox(
                "Select AI model:",
                options=models_available,
                index=models_available.index(current_model_name) if current_model_name in models_available else 0,
                key="main_model_selector"
            )
        
        with col2:
            # Button to switch models
            if st.button("Switch Model", key="main_switch_model"):
                new_model_id = model_mapping[new_model]
                if new_model_id != st.session_state.chatbot.model_name:
                    st.session_state.chatbot.set_model(new_model_id)
                    st.success(f"Switched to {new_model}")
                    safe_rerun()
        
        with col3:
            # Clear chat button
            if st.button("Clear Chat", key="main_clear_chat"):
                st.session_state.chatbot.clear_history()
                st.session_state.messages = []
                st.success("Chat history cleared")
                safe_rerun()
    
    # PDF upload capability in the main area with multiple file support
    if st.session_state.chatbot:
        with st.expander("📄 Upload PDF Documents", expanded=False):
            main_uploaded_files = st.file_uploader(
                "Choose PDF files to chat with", 
                type=["pdf"], 
                accept_multiple_files=True,
                key="main_pdf_uploader"
            )
            
            if main_uploaded_files:
                if st.button("Process PDFs", key="main_process_pdfs"):
                    for pdf_file in main_uploaded_files:
                        st.session_state.chatbot.load_pdf(pdf_file)
        
        # Show currently loaded PDFs with ability to remove them
        if hasattr(st.session_state, 'loaded_pdfs') and st.session_state.loaded_pdfs:
            with st.expander("📚 Loaded Documents", expanded=True):
                st.write(f"Currently loaded documents: {len(st.session_state.loaded_pdfs)}")
                
                for pdf_name in st.session_state.loaded_pdfs:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.write(f"• {pdf_name}")
                    with col2:
                        if st.button("Remove", key=f"remove_{pdf_name}"):
                            st.session_state.chatbot.remove_pdf(pdf_name)
                            safe_rerun()
    
    # Display warning if chatbot is not initialized
    if not st.session_state.chatbot:
        st.warning("Please initialize the chatbot from the sidebar first.")
    else:
        # PDF status indicator
        if st.session_state.chatbot.pdf_loaded:
            st.success(f"{len(st.session_state.loaded_pdfs)} PDFs loaded and ready for questions")
        else:
            st.info("No PDFs loaded yet. You can still chat in general mode.")
            
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")
        
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