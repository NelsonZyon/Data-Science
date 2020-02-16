# Árvore de decisão é um gráfico para representar opções e seus resultados na forma de uma árvore.
# Os nós no gráfico representam um evento ou escolha e as bordas do gráfico representam as regras ou 
# condições de decisão. É usado principalmente em aplicativos Machine Learning e Data Mining usando R.
# 
# Exemplos de uso de árvores de decisão são: prever um email como spam ou não spam, 
# prever um tumor é canceroso ou prever um empréstimo como um risco de crédito bom ou ruim, 
# com base nos fatores de cada um deles. Geralmente, um modelo é criado com dados observados
# também chamados de dados de treinamento. Em seguida, um conjunto de dados de validação é usado 
# para verificar e melhorar o modelo. R possui pacotes que são usados para criar e visualizar 
# árvores de decisão. Para um novo conjunto de variáveis preditivas, usamos esse modelo para chegar 
# a uma decisão sobre a categoria (sim / não, spam / não spam) dos dados.
# 
# O pacote R "party" é usado para criar árvores de decisão.
# 
# Instalar pacote R
# 
# Use o comando abaixo no console do R para instalar o pacote. 
# Você também precisa instalar os pacotes dependentes, se houver.

install.packages("party")

# O pacote "party" possui a função ctree() que é usada para criar e analisar a árvore de decisões.

# Sintaxe:

# A sintaxe básica para criar uma árvore de decisão em R é :

# ctree(formula, data)

# A seguir, é apresentada a descrição dos parâmetros usados:
# 
#   - formula é uma fórmula que descreve as variáveis preditoras e de resposta.
#   - data é o nome do conjunto de dados usado.


# DADOS DE ENTRADA

# Usaremos o conjunto de dados incorporado R chamado "readingSkills" para criar uma árvore de decisão. 
# Ele descreve a pontuação das habilidades de leitura de alguém, se conhecermos as variáveis 
# "age", "shoesize", "score" e se a pessoa é um falante nativo ou não.

# Aqui estão os dados de amostra.

# Carregue o pacote de festa. Carregará automaticamente outro
# pacotes dependentes.
library(party)

# Imprima alguns registros do conjunto de dados readingSkills.
print(head(readingSkills))

# Quando executamos o código acima, ele produz o seguinte resultado e gráfico -

# EXEMPLO
# Usaremos a função ctree() para criar a árvore de decisão e ver seu gráfico.

# Crie o quadro de dados de entrada.
input.dat <- readingSkills[c(1:105),]

# Dê um nome para o gráfico
png(file = "arvore_decisao.png")

# Crie a árvore.

output.tree <- ctree( nativeSpeaker ~ age + shoeSize + score, data = input.dat)


# Plote a árvore
plot(output.tree)

# Guarde o arquivo
dev.off()


# Conclusão

# A partir da árvore de decisão mostrada acima, podemos concluir que qualquer pessoa cuja 
# pontuação de habilidades de leitura seja inferior a 38,3 e a idade seja superior a 6 anos não é um 
# falante nativo.





