import streamlit as st
from graph import RagAgentGraph
from document_loader import DocumentProcessor
from vector_db import VectorDatabaseManager
from langsmith_evaluator import LangSmithEvaluator
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Agent - RAG Pipeline", layout="wide", page_icon="🤖")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    try:
        st.session_state.agent = RagAgentGraph()
        st.session_state.agent.build_graph()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        st.error(f"Failed to initialize agent: {str(e)}")
        st.stop()

if "evaluator" not in st.session_state:
    st.session_state.evaluator = LangSmithEvaluator()

# Header
st.title("🤖 AI Agent - RAG Pipeline")
st.markdown("*Powered by LangGraph, LangChain, and Azure OpenAI*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📄 Document Management")
    
    # Document upload section
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDF files to add to the knowledge base"
    )
    
    if uploaded_files:
        processor = DocumentProcessor()
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                try:
                    # # Save temporarily
                    # temp_path = f"/tmp/{uploaded_file.name}"
                    # print(temp_path)
                    # with open(temp_path, "wb") as f:
                    #     f.write(uploaded_file.getvalue())
                    import tempfile

                    # Get a temporary directory that exists on all systems
                    tempdir = tempfile.gettempdir()

                    temp_path = os.path.join(tempdir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

            

                    # Process
                    processor.load_pdf(temp_path)
                    st.success(f"✅ Processed: {uploaded_file.name}")
                    logger.info(f"Successfully processed: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    logger.error(f"Error processing {uploaded_file.name}: {e}")
    
    st.divider()
    
    # Vector DB statistics
    st.subheader("📊 Vector Database Stats")
    try:
        vector_db = VectorDatabaseManager()
        collection_info = vector_db.get_collection_info()
        
        if "error" in collection_info:
            st.info("Vector DB not initialized. Upload documents to initialize.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                doc_count = len(uploaded_files) if uploaded_files else 0
                st.metric("Documents", doc_count)
            with col2:
                st.metric("Vector Size", collection_info.get("vector_size", 0))
                
    except Exception as e:
        st.info("Vector DB not initialized")
        logger.warning(f"Could not fetch vector DB stats: {e}")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    
    show_metadata = st.checkbox("Show Query Metadata", value=True)
    show_evaluation = st.checkbox("Show LangSmith Evaluation", value=True)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.subheader("💬 Chat Interface")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available and enabled
        if show_metadata and "metadata" in message and message["role"] == "assistant":
            with st.expander("📋 Query Metadata"):
                metadata = message["metadata"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Route", metadata.get("route", "N/A"))
                with col2:
                    st.metric("Status", metadata.get("status", "N/A"))
                with col3:
                    st.metric("Response Length", len(metadata.get("response", "")))
        
        # Show evaluation if available and enabled
        if show_evaluation and "evaluation" in message and message["role"] == "assistant":
            with st.expander("📊 LangSmith Evaluation"):
                eval_data = message["evaluation"]
                
                if "scores" in eval_data:
                    st.markdown("**Evaluation Scores:**")
                    for key, value in eval_data["scores"].items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace("_", " ").title(), f"{value:.2f}")
                        else:
                            st.text(f"{key.replace('_', ' ').title()}: {value}")
                
                if "overall_score" in eval_data:
                    st.metric("Overall Score", f"{eval_data['overall_score']:.2f}", 
                             help="Average of all evaluation metrics")

# Chat input
if prompt := st.chat_input("Ask a question about weather or documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Invoke agent
                result = st.session_state.agent.invoke(prompt)
                response = result.get("response", "No response generated")
                route = result.get("route", "unknown")
                
                # Display response
                st.markdown(response)
                
                # Evaluate response
                evaluation = st.session_state.evaluator.evaluate_response(
                    query=prompt,
                    response=response,
                    route=route
                )
                
                # Log to LangSmith
                try:
                    st.session_state.evaluator.log_to_langsmith(
                        query=prompt,
                        response=response,
                        route=route,
                        metadata=result
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to LangSmith: {e}")
                
                # Prepare metadata
                metadata = {
                    "route": route,
                    "status": "Success" if response and "error" not in response.lower() else "Error",
                    "response": response
                }
                
                # Show metadata if enabled
                if show_metadata:
                    with st.expander("📋 Query Metadata"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Route", metadata["route"])
                        with col2:
                            st.metric("Status", metadata["status"])
                        with col3:
                            st.metric("Response Length", len(response))
                
                # Show evaluation if enabled
                if show_evaluation and evaluation:
                    with st.expander("📊 LangSmith Evaluation"):
                        if "scores" in evaluation:
                            st.markdown("**Evaluation Scores:**")
                            for key, value in evaluation["scores"].items():
                                if isinstance(value, (int, float)):
                                    st.metric(key.replace("_", " ").title(), f"{value:.2f}")
                                else:
                                    st.text(f"{key.replace('_', ' ').title()}: {value}")
                        
                        if "overall_score" in evaluation:
                            st.metric("Overall Score", f"{evaluation['overall_score']:.2f}",
                                     help="Average of all evaluation metrics")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": metadata,
                    "evaluation": evaluation
                })
                
                logger.info(f"Successfully processed query: {prompt[:50]}...")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                logger.error(f"Chat error: {e}", exc_info=True)
                
                # Add error to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": {"status": "Error", "route": "unknown"}
                })

# Footer
st.markdown("---")
