import ollama
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from vectorstore import query_vectorstore
from dotenv import load_dotenv
from langfuse import observe, get_client

load_dotenv()
langfuse = get_client()

LLM_MODEL = "llama3.1:8b"
MAX_RETRIES = 2

# ---- Define the shared state that flows through the graph ----
class AgentState(TypedDict):
    question: str
    original_question: str
    documents: List[str]
    sources: List[str]
    generation: str
    retry_count: int

# ---- Node 1: Retrieve ----
@observe()
def retrieve(state: AgentState) -> AgentState:
    print(f"\n[RETRIEVE] Searching for: '{state['question']}'")
    results = query_vectorstore(state["question"], n_results=6)
    docs = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return {**state, "documents": docs, "sources": sources}

# ---- Node 2: Grade documents (this is the "agentic" judgement step) ----
@observe()
def grade_documents(state: AgentState) -> AgentState:
    print("[GRADE] Evaluating relevance of retrieved chunks...")
    relevant_docs = []
    relevant_sources = []

    for doc, source in zip(state["documents"], state["sources"]):
        prompt = f"""You are grading whether a document chunk directly helps answer a specific question.
Question: {state['original_question']}
Document chunk: {doc}

Does this chunk contain SPECIFIC information that helps answer this exact question?
Generic mentions of the same broad topic do not count as relevant.
Reply with only one word: "yes" or "no"."""

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        verdict = response["message"]["content"].strip().lower()

        if "yes" in verdict:
            relevant_docs.append(doc)
            relevant_sources.append(source)
            print(f"  ✓ Relevant chunk from {source}")
        else:
            print(f"  ✗ Rejected chunk from {source}")

    print("\n[DEBUG] Accepted chunks content:")
    for doc in relevant_docs:
        print(f" - {doc[:150]}...\n")

    return {**state, "documents": relevant_docs, "sources": relevant_sources}

# ---- Conditional edge: decide next step based on grading ----
def decide_next_step(state: AgentState) -> str:
    if len(state["documents"]) >= 2:
        return "generate"
    elif state["retry_count"] < MAX_RETRIES:
        return "transform_query"
    else:
        print("[DECISION] Max retries reached, generating with whatever we have.")
        return "generate"

# ---- Node 3: Transform query (reformulate if retrieval was poor) ----
@observe()
def transform_query(state: AgentState) -> AgentState:
    print(f"[REFORMULATE] Original retrieval was weak. Rewriting query (attempt {state['retry_count'] + 1})...")
    prompt = f"""The following question did not retrieve relevant results from a documentation search.
Rewrite it as a clearer, more specific search query, focused on the same intent.
IMPORTANT: "RAG" means "Retrieval-Augmented Generation" - do not expand or reinterpret this acronym as anything else.
Original question: {state['original_question']}

Reply with ONLY the rewritten query, nothing else."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    new_query = response["message"]["content"].strip()
    print(f"  New query: '{new_query}'")

    return {**state, "question": new_query, "retry_count": state["retry_count"] + 1}

# ---- Node 4: Generate final answer ----
@observe()
def generate(state: AgentState) -> AgentState:
    print("[GENERATE] Producing final answer...")
    context = "\n\n".join(state["documents"]) if state["documents"] else "No relevant context found."

    prompt = f"""Answer the question using ONLY the context provided below. 
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {state['original_question']}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return {**state, "generation": response["message"]["content"]}

# ---- Build the graph ----
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {"generate": "generate", "transform_query": "transform_query"}
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()

@observe(name="agentic-rag-run")
def run_agent(question: str):
    agent = build_agent()
    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "sources": [],
        "generation": "",
        "retry_count": 0
    }
    result = agent.invoke(initial_state)
    return result

# ---- Run it ----
if __name__ == "__main__":
    question = "What are the main components of an agentic RAG system?"
    result = run_agent(question)

    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(result["generation"])
    print(f"\nSources used: {set(result['sources'])}")

    langfuse.flush() # Ensures trace data is sent before script exits