import ollama

#Texto de exemplo inspirado na videoaula do professor
texto = "Rei e Rainha"

print(f"Gerando os embeddings locais para o texto: '{texto}'")

#Gerando a representaçãp numérica (embeddings) com o Ollama
resposta = ollama.embeddings(model='mxbai-embed-large', prompt=texto)

#Gardando o vetor gerado
vetor = resposta['embedding']

#Mostrando o resultado de forma clara
print("\n--- RESULTADO NA PRÁTICA ---")
print(f"Termo analisado: {texto}")
print(f"Total de dimensões do vetor: {len(vetor)}")
print(f"Primeiros valores numéricos gerados:\n{vetor[:4]}")