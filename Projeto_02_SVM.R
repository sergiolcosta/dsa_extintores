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

###Importação dos Pacotes
library(readxl)
library(ggplot2)
library(corrplot)
library(caret)
library(e1071)


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

str(df_xlsx)

summary(df_xlsx)

View(df_xlsx)

### Criando um histograma com todas as variáveis do Dataset ###
# Loop sobre todas as colunas do dataset
for (col in colnames(df_xlsx)) {
  
  # Verificar se a coluna é numérica ou fator
  if (is.numeric(df_xlsx[[col]]) || is.factor(df_xlsx[[col]])) {
    
    # Criar o dataframe com a contagem de ocorrências
    df_count <- as.data.frame(table(df_xlsx[[col]]))
    
    # Criar o gráfico
    ggplot(df_count, aes(x = df_count[[1]], y = df_count[[2]])) +
      geom_bar(stat = "identity") +
      labs(title = paste("Histograma da variável", col),
           x = col,
           y = "Frequência") -> grafico
    
    # Mostrar o gráfico na tela
    print(grafico)
  }
}

# Convertendo os dados da coluna Fuel de texto em números e gravando nela mesmo.
df_xlsx$FUEL <- ifelse(df_xlsx$FUEL == "gasoline", 1, 
                           ifelse(df_xlsx$FUEL == "kerosene", 2,
                                  ifelse(df_xlsx$FUEL == "lpg", 3, 
                                         ifelse(df_xlsx$FUEL == "thinner", 4, NA))))

# Convertendo a coluna "FUEL" em fator
df_xlsx$FUEL <- factor(df_xlsx$FUEL)

View(df_xlsx)

str(df_xlsx)


######################## Criando mapa de correlação ############################

# Criando um novo dataset e convertendo todas as colunas para numéricas
df_corr <- data.frame(sapply(df_xlsx, as.numeric))

# Selecionar todas as colunas exceto a coluna STATUS
df_corr <- df_corr[, -7]
#View(df_corr)
#View(df_xlsx)

# Criar a matriz de correlação e verifica em números
corr_matrix <- cor(df_corr)
corr_matrix

# Plotar o gráfico de correlação
corrplot(corr_matrix, type = "full", order = "hclust", tl.col = "black", tl.srt = 45)


#------------------------------------------------------

############################ MACHINE LEARNING ##################################
########## Modelo SVM - Linear / Radial / Polynomial / Sigmoid #################

# Convertendo todas as variáveis para numéricas
df_norm <- as.data.frame(sapply(df_xlsx, as.numeric))

# Escalando todas as variáveis, exceto STATUS
df_norm[, -7] <- scale(df_norm[, -7])

# Convertendo a coluna STATUS para fator
df_norm$STATUS <- as.factor(df_norm[, "STATUS"])

# Criando índices de treino e teste
set.seed(123)
trainIndex <- createDataPartition(as.factor(df_norm[, "STATUS"]), p = .7, 
                                  list = FALSE, times = 1)

df_train <- df_norm[trainIndex, ]
df_test <- df_norm[-trainIndex, ]

# Modelos SVM com diferentes kernels - Linear / Radial / Polynomial / Sigmoid
svm_models <- list(
  svm_linear = svm(STATUS ~ ., data = df_train, kernel = "linear"),
  svm_radial = svm(STATUS ~ ., data = df_train, kernel = "radial"),
  svm_poly = svm(STATUS ~ ., data = df_train, kernel = "polynomial"),
  svm_sig = svm(STATUS ~ ., data = df_train, kernel = "sigmoid")
)

# Loop pelos modelos SVM para calcular a acurácia e matriz de confusão
accuracy <- c()
for (i in seq_along(svm_models)) {
  # Fazer previsões com o modelo SVM
  svm_pred <- predict(svm_models[[i]], df_test[, -which(names(df_test) == "STATUS")])
  
  # Calcular a acurácia do modelo
  svm_acc <- mean(svm_pred == df_test[, "STATUS"])
  accuracy[i] <- svm_acc
  cat(sprintf("Acurácia do modelo SVM (%s): %.2f%%\n", names(svm_models)[i], svm_acc*100))
  
  # Fazer as previsões com o modelo SVM nos dados de teste
  svm_pred <- predict(svm_models[[i]], newdata = df_test)
  
  # Calcular a matriz de confusão e as métricas de avaliação
  confusion_mat <- confusionMatrix(svm_pred, df_test[, "STATUS"])
  cat(sprintf("Matriz de Confusão do modelo SVM (%s):\n", names(svm_models)[i]))
  print(confusion_mat$table)
  cat("\n")
}

# Plotar gráfico de barras com as acurácias dos modelos SVM
barplot(accuracy * 100, names.arg = names(svm_models), ylim = c(0, 100),
        col = rainbow(length(svm_models)))
title(main = "Acurácias dos modelos SVM")



######################### MODELO COM A MELHOR ACURÁCIA #########################
############################# MODELO SVM - Radial ##############################

# Convertendo todas as variáveis para numéricas
df_norm <- as.data.frame(sapply(df_xlsx, as.numeric))

# Escalando todas as variáveis, exceto STATUS
df_norm[, -7] <- scale(df_norm[, -7])

View(df_norm)

# Convertendo a coluna STATUS para fator
df_norm$STATUS <- as.factor(df_norm[, "STATUS"])

# Criando índices de treino e teste
set.seed(123)
trainIndex <- createDataPartition(as.factor(df_norm[, "STATUS"]), p = .7, 
                                  list = FALSE, times = 1)

df_train <- df_norm[trainIndex, ]
df_test <- df_norm[-trainIndex, ]

# Modelo SVM com kernel radial
svm_model <- svm(STATUS ~ ., data = df_train, kernel = "radial")

# Fazer previsões com o modelo SVM nos dados de teste
svm_pred <- predict(svm_model, newdata = df_test)

# Calcular a acurácia do modelo
svm_acc <- mean(svm_pred == df_test[, "STATUS"])
cat(sprintf("Acurácia do modelo SVM (radial): %.2f%%\n", svm_acc*100))

# Calcular a matriz de confusão e as métricas de avaliação
confusion_mat <- confusionMatrix(svm_pred, df_test[, "STATUS"])
print(confusion_mat$table)

