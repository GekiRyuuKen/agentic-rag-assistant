import streamlit as st
from agent import build_agent, run_agent

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Agentic RAG Documentation Assistant")
st.caption("Ask questions about RAG and Agentic RAg - backed by 4 research survey papers, with self-correction retrieval.")


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources used"):
                for source in msg["sources"]:
                    st.write(f"- {source}")

# Chat input
question = st.chat_input("Ask a question about RAG or Agentic RAG...")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Run the agent with a spinner while it works
    with st.chat_message("assistant"):
        with st.spinner("Thinking - retrieving, grading, and generating..."):
            result = run_agent(question)
            answer = result["generation"]
            sources = list(set(result["sources"]))

        st.write(answer)
        if sources:
            with st.expander("Sources used"):
                for source in sources:
                    st.write(f"- {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })