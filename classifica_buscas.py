# -*- coding: utf-8 
#O algoritmo base(algoritmo burro) precisa ser testado sempre com o elemento que mais aparece, ou 1 ou 0  e etc...
from collections import Counter
import pandas as pd

# teste inicial: home, busca, logado => comprou
# home, busca
# home, logado
# busca, logado
# busca: 85.71% (7 testes) Esse resultado vai servrir para mostrar no relatório.

#Data frame ser lido
df = pd.read_csv('busca.csv')

#separa o data frame
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']


#Processou os dammies, apenas para valores qualitativo
Xdummies_df = pd.get_dummies(X_df) # São variaveis que a gente colocar para preencher o espaço de outro qur a gente tinh feito, por exemplo 'Algoritmo'
Ydummies_df = Y_df

#Pega os valores do dataFrame e transforma em array
X = Xdummies_df.values
Y = Ydummies_df.values

#Treinado e Testando meu algoritimo

#90% de treino 10% de teste
tamanho_de_treino = int(0.9 * len(Y)) #900 90%	
tamanho_de_teste  = int(len(Y) - tamanho_de_treino) #100

#90% iniciais
treino_dados = X[:tamanho_de_treino] # 900 primeiros
treino_marcacoes = Y[:tamanho_de_treino] #100

#10% finais
teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

# print(treino_dados)

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

tamanho_de_treino = porcentagem_de_treino * len(Y)
tamanho_de_teste = porcentagem_de_teste * len(Y)
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

fim_de_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]




def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
	modelo.fit(treino_dados, treino_marcacoes)

	resultado = modelo.predict(teste_dados)
	acertos = (resultado == teste_marcacoes)

	total_de_acertos = sum(acertos)
	total_de_elementos = len(teste_dados)
	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
	print(msg)
	return taxa_de_acerto


from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)


from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomial > resultadoAdaBoost:
	vencedor = modeloMultinomial
else:
	vencedor = modeloAdaBoost

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
print(msg)



# a eficacia do algoritmo que chuta
# tudo um unico valor
acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d " % len(validacao_dados))






