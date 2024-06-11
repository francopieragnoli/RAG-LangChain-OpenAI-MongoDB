from pymongo import MongoClient
from langchain.embeddings import LangChainEmbeddings
from PyPDF2 import PdfFileReader
import os

# Conectar ao MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['vetorial_db']
collection = db['vetores']

# Função para extrair texto de arquivos PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfFileReader(file)
        text = ''
        for page_num in range(reader.getNumPages()):
            text += reader.getPage(page_num).extract_text()
    return text

# Função para dividir o texto em chunks coerentes de até 256 caracteres
def split_text_into_chunks(text, chunk_size=256):
    sentences = text.split('.')
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + '.'
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + '.'
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Carregar embeddings do LangChain
embeddings = LangChainEmbeddings()

# Caminho para o diretório de PDFs
pdf_directory = '/path/to/pdf/files/'

# Processar cada PDF e armazenar os vetores no MongoDB
for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            vector = embeddings.embed_text(chunk)
            collection.insert_one({'filename': filename, 'chunk': chunk, 'vector': vector})

print("Banco vetorial criado com sucesso.")
