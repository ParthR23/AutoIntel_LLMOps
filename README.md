# 🚗 AutoIntel AI: Agentic Service Technician

**AutoIntel AI** is a state-of-the-art, agentic RAG (Retrieval-Augmented Generation) system designed for the 2026 automotive industry. It combines specialized technical knowledge from vehicle manuals with real-time safety data via official APIs.

## 🌟 Key Features

- **Agentic Routing:** Uses LangGraph to intelligently decide between consulting internal manuals (Pinecone) or fetching live recall data (NHTSA API).
- **Structured Extraction:** Automatically parses vehicle Make/Model/Year from natural language queries.
- **Self-Correction Grader:** A built-in "Honesty Loop" that forces the agent to say "I don't know" if the information isn't found in the manuals, preventing hallucinations.
- **Safety Guardrails:** Integrated with **Llama Guard 3** to filter out harmful or dangerous technical advice.
- **LLMOps Ready:** Full observability with **LangSmith** and automated quality scoring using the **Ragas** framework.

## 🏗️ Architecture



## 🛠️ Tech Stack

- **LLM:** Llama-3.3-70b (via Groq)
- **Orchestration:** LangGraph (Stateful Agents)
- **Vector DB:** Pinecone (Serverless)
- **Safety:** Llama Guard 3
- **Evaluation:** Ragas (Faithfulness, Relevancy, Precision)
- **UI:** Streamlit

## 🚀 Quick Start

1. **Clone & Install:**
   ```bash
   git clone [https://github.com/yourusername/autointel-ai.git](https://github.com/yourusername/autointel-ai.git)
   cd autointel-ai
   pip install -r requirements.txt

2. **Demo Video:**
   https://www.loom.com/share/87f674e9489b4afdab03bfff8e729db7