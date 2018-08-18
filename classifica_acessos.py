# -*- coding: utf-8 
from dados import carregar_acessos
X,Y = carregar_acessos()
#ETAPA 1 : PRÉ-PROCESSAMENTO E VISUALIZAÇÃO DOS DADOS.
print("PRÉ PROCESSAMENTO E VISUALIZAÇÃO DOS DADOS!")
print("VALORES DE X---------------------------------------------------------X")
print(X)
print("VALORES DE Y---------------------------------------------------------Y")
print(Y)

# 1. separar 90% para treino e 10% para teste: 88.89%
treino_dados = X[:90]
treino_marcacoes = Y[:90]	#Marcações é o resultado se comprou ou não.

teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

#Algoritmo neyve bayes para fazer classificação.
from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

#Predict serve para prever os resultados que estão sendo treinados, que esao no scv
resultado = modelo.predict(teste_dados) #Resultado é o que eu previ, se vai comprar(0) ou não vai comprar(0)
diferencas = resultado - teste_marcacoes # Se a diferença for 0, significa que o modelo acertou, ou seja, comprou. É 0.
 
 #Fazendo o for para calcular os acertos.
acertos = [d for d in diferencas if d == 0] # os que forem igual a 0 sao os que comprou
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

#Imprimindo resultados.
print(taxa_de_acerto)
print(total_de_elementos)




