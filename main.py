from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # IMPORT CORRETO
from langchain_community.vectorstores import FAISS

# Conectando ao modelo local Ollama para geração de texto
llm = Ollama(model="gemma3:12b")
memory = ConversationBufferMemory()

# Carrega e divide o PDF
loader = PyPDFLoader("meudoc.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Usa embeddings do Hugging Face (local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cria base vetorial FAISS com esses embeddings
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Prompt com contexto (documento) + histórico
prompt_template = PromptTemplate(
    input_variables=["context", "history", "input"],
    template="""
Você é um assistente inteligente que responde com base em um documento e no histórico da conversa. Fale em português e seja direto.

Contexto do documento:
{context}

Histórico da conversa:
{history}

Usuário: {input}
IA:"""
)

# Loop de chat com RAG
print("🤖 Olá! Sou um assistente com base em documento. Digite 'sair' para encerrar.\n")

while True:
    user_input = input("Você: ")
    if user_input.strip().lower() == "sair":
        print("IA: Até mais! 👋")
        break

    # Recupera contexto relevante do PDF
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])

    # Prepara prompt com histórico + contexto
    prompt_text = prompt_template.format(
        context=context,
        history=memory.buffer,
        input=user_input
    )

    # Streaming da resposta
    print("IA: ", end="", flush=True)
    response = ""
    for chunk in llm.stream(prompt_text):
        print(chunk, end="", flush=True)
        response += chunk
    print()

    # Atualiza memória
    memory.save_context({"input": user_input}, {"output": response})
