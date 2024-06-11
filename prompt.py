from pymongo import MongoClient
from langchain.embeddings import LangChainEmbeddings
import numpy as np
import openai

# Configurar a API da OpenAI
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Conectar ao MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['vetorial_db']
collection = db['vetores']

# Definir o nível de assertividade em percentual
ASSERTIVENESS_THRESHOLD = 0.7  # 70%

# Função para calcular similaridade de cosseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Carregar embeddings do LangChain
embeddings = LangChainEmbeddings()

# Função para buscar documentos relevantes
def search_documents(prompt):
    prompt_vector = embeddings.embed_text(prompt)
    results = collection.find()
    similarities = []
    
    for result in results:
        vector = result['vector']
        similarity = cosine_similarity(prompt_vector, vector)
        similarities.append((result['filename'], result['chunk'], similarity))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities

# Função para obter resposta da OpenAI
def get_openai_response(prompt, context):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Contexto: {context}\n\nPergunta: {prompt}\n\nResposta:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Consulta de exemplo
prompt = "Digite sua pergunta aqui"
results = search_documents(prompt)

# Filtrar chunks acima do nível de assertividade
top_chunks = " ".join([chunk for _, chunk, similarity in results if similarity >= ASSERTIVENESS_THRESHOLD])

# Obter a resposta final da OpenAI
if top_chunks:
    final_response = get_openai_response(prompt, top_chunks)
else:
    final_response = "Desculpe, não foi possível encontrar informações relevantes para sua pergunta."

print("Resposta final:")
print(final_response)

print("Consulta realizada com sucesso.")
