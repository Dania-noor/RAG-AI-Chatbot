from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# LOAD CHROMADB
# =========================

vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# =========================
# LOCAL LLM
# =========================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# =========================
# QUERY UNDERSTANDING PROMPT
# =========================

query_prompt = ChatPromptTemplate.from_template("""
You are a query understanding assistant.

The user may type with:
- spelling mistakes
- missing letters
- extra letters
- informal language
- abbreviations
- missing punctuation

Your job is to understand the user's intended question.

IMPORTANT:
- Preserve the user's original meaning.
- Do NOT change technical terms into unrelated words.
- Preserve acronyms such as RAG, AI, LLM, NLP, API, etc.
- If the user writes "ragg" and the context suggests "RAG",
  interpret it as RAG.
- Do NOT answer the question.
- Return ONLY the corrected/intended question.
- If the question is already clear, return it unchanged.

User question:
{question}
""")


# =========================
# CONVERSATION INTENT PROMPT
# =========================

intent_prompt = ChatPromptTemplate.from_template("""
Classify the user's message into exactly ONE category:

GREETING
THANKS
GOODBYE
QUESTION

Examples:

"hi" -> GREETING
"hellooo" -> GREETING
"assalam o alaikum" -> GREETING

"thanku" -> THANKS
"thnks a lot" -> THANKS
"thank you so much" -> THANKS

"bye" -> GOODBYE
"allah hafiz" -> GOODBYE

"What is RAG?" -> QUESTION
"wht is machine learning" -> QUESTION
"tell me about embeddings" -> QUESTION

Return ONLY one word:
GREETING, THANKS, GOODBYE, or QUESTION

User message:
{question}
""")


# =========================
# ANSWER PROMPT
# =========================

answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Rules:

1. Answer the user's question using ONLY the provided
   document context.

2. If the answer is not available in the context, say:
   "I don't have enough information in the provided documents."

3. Never invent or assume information.

4. Keep the answer clear and concise.

5. Conversation history can be used to understand follow-up
   questions, but factual answers must come from the document
   context.

Conversation History:
{history}

Document Context:
{context}

Question:
{question}

Answer:
""")


# =========================
# CONVERSATION RESPONSES
# =========================

conversation_responses = {
    "GREETING": "Hello! How can I help you?",
    "THANKS": "You're welcome! 😊",
    "GOODBYE": "Goodbye! Have a great day!"
}


# =========================
# CONVERSATION HISTORY
# =========================

history = []


# =========================
# START CHATBOT
# =========================

print("=" * 60)
print("RAG AI CHATBOT")
print("=" * 60)

print("Type 'exit' to stop the chatbot.")


# =========================
# CHAT LOOP
# =========================

while True:

    question = input("\nYou: ")

    # =========================
    # EXIT
    # =========================

    if question.lower().strip() == "exit":

        print("\nChatbot stopped.")
        break


    # =========================
    # UNDERSTAND USER INTENT
    # =========================

    intent_messages = intent_prompt.format_messages(
        question=question
    )

    intent = llm.invoke(intent_messages).content.strip().upper()


    # =========================
    # HANDLE CONVERSATION
    # =========================

    if intent in conversation_responses:

        response_text = conversation_responses[intent]

        print("\nBot:", response_text)

        history.append(f"User: {question}")
        history.append(f"Assistant: {response_text}")

        continue


    # =========================
    # UNDERSTAND / CORRECT QUERY
    # =========================

    query_messages = query_prompt.format_messages(
        question=question
    )

    corrected_question = llm.invoke(query_messages).content.strip()


    # =========================
    # RETRIEVAL
    # =========================

    print("\nSearching documents...")

    results = vectorstore.similarity_search(
        corrected_question,
        k=3
    )


    # =========================
    # CREATE CONTEXT
    # =========================

    context = "\n\n".join(
        document.page_content
        for document in results
    )


    # =========================
    # HISTORY
    # =========================

    history_text = "\n".join(history[-10:])


    # =========================
    # GENERATE ANSWER
    # =========================

    messages = answer_prompt.format_messages(
        history=history_text,
        context=context,
        question=corrected_question
    )

    response = llm.invoke(messages)


    # =========================
    # DISPLAY ANSWER
    # =========================

    print("\nBot:", response.content)


    # =========================
    # SAVE HISTORY
    # =========================

    history.append(f"User: {question}")
    history.append(f"Assistant: {response.content}")