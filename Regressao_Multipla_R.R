# A regressão múltipla é uma extensão da regressão linear na relação entre mais de duas variáveis.
# Na relação linear simples, temos uma variável preditora e uma resposta, mas, na regressão múltipla,
# temos mais de uma variável preditora e uma variável de resposta.


# A equação matemática geral para regressão múltipla é :
# y = a + b1x1 + b2x2 +...bnxn

# A seguir, é apresentada a descrição dos parâmetros usados
# 
#  > y é a variável de resposta.
#  > a, b1, b2 ... bn são os coeficientes.
#  > x1, x2, ... xn são as variáveis preditoras

# Criamos o modelo de regressão usando a função lm() em R. O modelo determina o valor dos coeficientes 
# usando os dados de entrada. Em seguida, podemos prever o valor da variável de resposta para um determinado 
# conjunto de variáveis preditoras usando esses coeficientes.

# lm() Function
# Essa função cria o modelo de relacionamento entre o preditor e a variável de resposta.
# A sintaxe básica da função lm() na regressão múltipla é 

#   > lm(y ~ x1+x2+x3...,data)
# 
# A seguir, é apresentada a descrição dos parâmetros usados
# 
# formula é um símbolo que apresenta a relação entre a variável de resposta e as variáveis preditoras.
# 
# data é o vetor no qual a fórmula será aplicada.

# EXEMPLO

# DADOS DE ENTRADA

# Considere o conjunto de dados "mtcars" disponível no ambiente R. 
# Ele fornece uma comparação entre diferentes modelos de carros em termos de quilometragem por galão (mpg), 
# cilindrada ("disp"), potência ("hp"), peso do carro ("wt") e mais alguns parâmetros.


# O objetivo do modelo é estabelecer a relação entre "mpg" como uma variável de resposta com "disp", "hp" e "wt" 
# como variáveis preditoras. Criamos um subconjunto dessas variáveis a partir do conjunto de dados mtcars 
# para essa finalidade.

input <- mtcars[,c("mpg","disp","hp","wt")]
print(head(input))

# Quando executamos o código acima, ele produz o seguinte resultado 

# Crie um modelo de relacionamento e obtenha os coeficientes


input <- mtcars[,c("mpg","disp","hp","wt")]

# Crie um modelo de relacionamento 
model <- lm(mpg~disp+hp+wt, data = input)

# Exibe o modelo.
print(model)

# # Obtenha o Intercepto e os coeficientes como elementos vetoriais.
cat("# # # # Os valores do coeficiente # # # ","\n")

a <- coef(model)[1]
print(a)

Xdisp <- coef(model)[2]
Xhp <- coef(model)[3]
Xwt <- coef(model)[4]

print(Xdisp)
print(Xhp)
print(Xwt)

# Quando executamos o código acima, ele produz o seguinte resultado 

# Criar equação para o modelo de regressão
# Com base nos valores de interceptação e coeficiente acima, criamos a equação matemática.

# Y = a+Xdisp.x1+Xhp.x2+Xwt.x3
# or
# Y = 37.15+(-0.000937)*x1+(-0.0311)*x2+(-3.8008)*x3

# Aplicar equação para prever novos valores
# Podemos usar a equação de regressão criada acima para prever a milhagem quando um novo conjunto de 
# valores para deslocamento, potência e peso é fornecido.

# Para um carro com disp = 221, hp = 102 e peso = 2,91, a milhagem prevista é 

# Y = 37.15+(-0.000937)*221+(-0.0311)*102+(-3.8008)*2.91 = 22.7104



















