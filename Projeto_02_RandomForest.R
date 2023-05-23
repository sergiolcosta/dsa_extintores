################### PROJETO COM FEEDBACK com Feedback 2 ########################
############## Machine Learning em Aprendizagem Supervisionada #################
########### Prevendo eficiência de extintores na extinção de fogo ##############

#### Arquivos com informações. Projeto2.docx #####

# Configurando e checando a pasta de trabalho
setwd("/Users/ymportz/Desktop/FCD/ProjetosComFeedback/Projeto_2/")
getwd()

###Instalação dos Pacotes
#install.packages("readxl")
#install.packages("ggplot")
#install.packages("corrplot")
#install.packages("caret")
#install.packages("e1071")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("magrittr")
#install.packages("randomForest")


###Importação dos Pacotes
library(readxl)
library(ggplot2)
library(corrplot)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(magrittr)
library(randomForest)


# Carregando o arquivo .xlsx
df_xlsx <- read_excel("dados/Acoustic_Extinguisher_Fire_Dataset.xlsx")


### Analise Exploratória ###
# Verificando se existe algum valor N/A no dataset.
sum(is.na(df_xlsx))

str(df_xlsx)
summary(df_xlsx)


# Verificando se existem linhas duplicadas no conjunto de dados
if (any(duplicated(df_xlsx))) {
  cat("Existem linhas duplicadas no conjunto de dados.")
} else {
  cat("Não existem linhas duplicadas no conjunto de dados.")
}


# Convertendo a variável Size, Fuel e Status para fator
df_xlsx$SIZE <- as.factor(df_xlsx$SIZE)
df_xlsx$FUEL <- as.factor(df_xlsx$FUEL)
df_xlsx$STATUS <- as.factor(df_xlsx$STATUS)

str(df_xlsx)

summary(df_xlsx)

View(df_xlsx)


# Convertendo os dados da coluna Fuel de texto em números e gravando nela mesmo.
df_xlsx$FUEL <- ifelse(df_xlsx$FUEL == "gasoline", 1, 
                           ifelse(df_xlsx$FUEL == "kerosene", 2,
                                  ifelse(df_xlsx$FUEL == "lpg", 3, 
                                         ifelse(df_xlsx$FUEL == "thinner", 4, NA))))

# Convertendo a coluna "FUEL" em fator
df_xlsx$FUEL <- factor(df_xlsx$FUEL)

View(df_xlsx)

str(df_xlsx)



############################ MACHINE LEARNING #################################.
############################## RANDON FOREST ##################################.
##################### SEM OTIMIZAÇÀO DE HIPERPARAMETROS #######################.

# Dividindo os dados em treino e teste
set.seed(123)
train_index <- createDataPartition(df_xlsx$STATUS, p = 0.7, list = FALSE, times = 1)
train_data <- df_xlsx[train_index, ]
test_data <- df_xlsx[-train_index, ]

# Criando o modelo de árvore de decisão
tree_model <- rpart(STATUS ~ ., data = train_data, method = "class")

# Fazendo previsões com o modelo de árvore de decisão
pred_tree <- predict(tree_model, newdata = test_data, type = "class")

# Calculando a acurácia e a matriz de confusão
#options(digits = 4)

accuracy_tree <- mean(pred_tree == test_data$STATUS)
cat(sprintf("Acurácia do modelo de Árvore de Decisão: %.2f%%\n", accuracy_tree*100))
confusion_mat <- confusionMatrix(pred_tree, test_data$STATUS)

# Exibindo a matriz de confusão
cat("Matriz de Confusão do modelo de Árvore de Decisão:\n\n",
    paste(capture.output(print(confusion_mat$table)), collapse="\n"), "\n")

?rpart



############################ MACHINE LEARNING #################################.
############################## RANDON FOREST ##################################.
################## maxdepth = 5, minsplit = 20, cp = 0.005 ####################.

# Dividindo os dados em treino e teste
set.seed(123)
train_index <- createDataPartition(df_xlsx$STATUS, p = 0.7, list = FALSE, times = 1)
train_data <- df_xlsx[train_index, ]
test_data <- df_xlsx[-train_index, ]

# Criando o modelo de árvore de decisão com os hiperparâmetros ajustados
tree_model <- rpart(STATUS ~ ., data = train_data, method = "class", maxdepth = 5, minsplit = 20, cp = 0.005)

# Fazendo previsões com o modelo de árvore de decisão
pred_tree <- predict(tree_model, newdata = test_data, type = "class")

# Calculando a acurácia e a matriz de confusão
#options(digits = 4)

accuracy_tree <- mean(pred_tree == test_data$STATUS)
cat(sprintf("Acurácia do modelo de Árvore de Decisão: %.2f%%\n", accuracy_tree*100))
confusion_mat <- confusionMatrix(pred_tree, test_data$STATUS)

# Exibindo a matriz de confusão
cat("Matriz de Confusão do modelo de Árvore de Decisão:\n\n",
    paste(capture.output(print(confusion_mat$table)), collapse="\n"), "\n")



############################ MACHINE LEARNING #################################.
############################## RANDON FOREST ##################################.
################## maxdepth = 10, minsplit = 10, cp = 0.001 ####################.

# Dividindo os dados em treino e teste
set.seed(123)
train_index <- createDataPartition(df_xlsx$STATUS, p = 0.7, list = FALSE, times = 1)
train_data <- df_xlsx[train_index, ]
test_data <- df_xlsx[-train_index, ]

# Criando o modelo de árvore de decisão com os hiperparâmetros ajustados
tree_model <- rpart(STATUS ~ ., data = train_data, method = "class", maxdepth = 10, minsplit = 10, cp = 0.001)

# Fazendo previsões com o modelo de árvore de decisão
pred_tree <- predict(tree_model, newdata = test_data, type = "class")

# Calculando a acurácia e a matriz de confusão
#options(digits = 4)

accuracy_tree <- mean(pred_tree == test_data$STATUS)
cat(sprintf("Acurácia do modelo de Árvore de Decisão: %.2f%%\n", accuracy_tree*100))
confusion_mat <- confusionMatrix(pred_tree, test_data$STATUS)

# Exibindo a matriz de confusão
cat("Matriz de Confusão do modelo de Árvore de Decisão:\n\n",
    paste(capture.output(print(confusion_mat$table)), collapse="\n"), "\n")


