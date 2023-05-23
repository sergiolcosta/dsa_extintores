################### PROJETO COM FEEDBACK com Feedback 1 ########################
################# Machine Learning em Logística Prevendo o #####################
################## Consumo de Energia de Carros Elétricos ######################

#### Arquivos com informações. LeiaMe_Info.docx #####


setwd("/Users/ymportz/Desktop/FCD/ProjetosComFeedback/Projeto_1/")
getwd()


# INSTALAÇÃO DOS PACOTES
#install.packages("readxl")
#install.packages("haven")
#install.packages("ggplot")
#install.packages("corrplot")
#install.packages("tidyverse")
#install.packages("caret")
#install.packages("randomForest")

# IMPORTAÇÃO DOS PACOTES
library(readxl)
library(haven)
library(ggplot2)
library(dplyr)
library(corrplot)
library(tidyverse)
library(tidyr)
require(lattice)
library(glmnet)
library(caret)
library(randomForest)

# Importando arquivo .xlsx
df <- read_excel("Dados/FEV-data-Excel.xlsx")
#View(df)

#Mudando nomes de variáveis
colnames(df) <- c("CarFullModel", "Make", "Model", "MinimalPriceGrossPLN", "EnginePowerKM", "MaximumTorqueNM", "TypeOfBrakes", "DriveType", 
                  "BatteryCapacityKWH", "RangeWLTPKM", "WheelBaseCM", "LengthCM", "WidthCM", "HeightCM", 
                  "MinimalEmptyWeight", "PermissableGrossWeightKG", "MaximumLoadCapacity", "NumberOfSeats", 
                  "NumberOfDoors", "TireSizeIN", "MaximumSpeedKPH", "BootCapacityVDA", "Acceleration0_100KPHs", 
                  "MaximumDcChargingPower", "MeanEnergyConsumptionKWH100km")

# Adicionando "front + rear" na variável "TypeOfBrakes"
df[52, "TypeOfBrakes"] <- "disc (front + rear)"

# Preenchendo os valores N/A das outras variáveis com o valor médio da váriavel
df$PermissableGrossWeightKG <- replace_na(df$PermissableGrossWeightKG, mean(df$PermissableGrossWeightKG, na.rm = TRUE))
df$MaximumLoadCapacity <- replace_na(df$MaximumLoadCapacity, mean(df$MaximumLoadCapacity, na.rm = TRUE))
df$BootCapacityVDA <- replace_na(df$BootCapacityVDA, mean(df$BootCapacityVDA, na.rm = TRUE))
df$Acceleration0_100KPHs <- replace_na(df$Acceleration0_100KPHs, mean(df$Acceleration0_100KPHs, na.rm = TRUE))
df$MeanEnergyConsumptionKWH100km <- replace_na(df$MeanEnergyConsumptionKWH100km, mean(df$MeanEnergyConsumptionKWH100km, na.rm = TRUE))

View(df)

# Verificando se existe valores N/A
sum(is.na(df))
colSums(is.na(df))

# verificando a quantidade de linhas (53) e Colunas (25)
nrow(df)
ncol(df)

# pegando os nomes da colunas
names(df)

# Verificando os tipos de dados do dataset
str(df)


# Mudando tipo de dados para fator
df$`TypeOfBreaks` <- as.factor(df$`TypeOfBrakes`)
df$`DriveType` <- as.factor(df$`DriveType`)
df$`NumberOfSeats` <- as.factor(df$`NumberOfSeats`)

# Criar dataframe com variáveis categóricas.
df_categ <- df[, c("CarFullModel", "Make", "Model", "TypeOfBrakes", "DriveType", "NumberOfSeats")]
df_categ[] <- lapply(df_categ, factor) # converter todas as colunas para fator
summary(df_categ)


# Criar dataframe com variáveis numéricas
df_num <- dplyr::select_if(df, is.numeric)
summary(df_num)


########################### Histograma ########################################
# Seleciona apenas as variáveis numéricas do dataframe df
df_numericas <- df[, sapply(df, is.numeric)]

# Define a variável target
target <- "MeanEnergyConsumptionKWH100km"

# Cria uma lista vazia para armazenar os histogramas
histogramas <- list()

# Loop para gerar um histograma para cada variável
for (col in colnames(df_numericas)) {
  # Exclui a variável target do loop
  if (col != target) {
    # Verifica se todos os dados são numéricos
    if (all(is.numeric(df[[col]])) && is.numeric(df[[target]])) {
      # Gera o histograma da variável em relação à variável target
      hist_data <- hist(df[[col]], main = paste0("Histograma - ", col),
                        xlab = col, ylab = "Frequência")
      # Salva o histograma na lista
      histogramas[[col]] <- hist_data
    } else {
      # Emite uma mensagem de aviso se algum dos dados não for numérico
      message(paste0("Variável ", col, " ou target não é numérica."))
      
    }
  }
}


################################################################################
########################### MAPA DE CORRELAÇÃO #################################

# Definindo as colunas para a análise de correlação #
cols <- c("MinimalPriceGrossPLN", "EnginePowerKM", "MaximumTorqueNM", "BatteryCapacityKWH",
          "RangeWLTPKM", "WheelBaseCM", "LengthCM", "WidthCM", "HeightCM", 
          "MinimalEmptyWeight", "PermissableGrossWeightKG", "MaximumLoadCapacity", 
          "NumberOfDoors", "TireSizeIN", "MaximumSpeedKPH", "BootCapacityVDA", "Acceleration0_100KPHs", 
          "MaximumDcChargingPower", "MeanEnergyConsumptionKWH100km")

df_clean <- na.omit(df[, c(cols)])


# Métodos de Correlação
# Pearson - coeficiente usado para medir o grau de relacionamento entre duas variáveis com relação linear

# Métodos de correlação
metodos <- "pearson"

# Aplicando os métodos de correlação com a função cor()
cors <- lapply(metodos, function(method) 
  (cor(df_clean[, cols], method = method)))

# Substituindo a diagonal da matriz de correlação por 1
for(i in 1:length(cors)){
  diag(cors[[i]]) <- 1
}

# Preprando o plot
plot.cors <- function(x, labs){
  plot(levelplot(x, 
                 main = paste("Plot de Correlação usando Método", labs),
                 scales = list(x = list(rot = 90), cex = 0.6)))
  # Criando uma paleta de cores vermelha e azul
  my_palette <- colorRampPalette(c("red", "white", "blue"))(100)
  
  # Plotando o heatmap com a nova paleta de cores
  levelplot(x, col.regions = my_palette, 
            main = paste("Plot de Correlação usando Método", labs),
            scales = list(x = list(rot = 90), cex = 0.6))
}

# Cria o Mapa de Correlação
Map(plot.cors, cors, metodos)


################################################################################
################### CRIAÇÃO DO MODELO DE MACHINE LEARNING ######################

# seleciona apenas as colunas numéricas
df_num <- df[, sapply(df, is.numeric)]

# normaliza os dados
df_norm <- as.data.frame(scale(df_num))


# Definindo as variáveis preditoras e target
vars <- c("EnginePowerKM", "BatteryCapacityKWH", "LengthCM", "PermissableGrossWeightKG", "MaximumDcChargingPower")
target <- "MeanEnergyConsumptionKWH100km"

# Criando conjunto de treino e teste
set.seed(123)
trainIndex <- createDataPartition(df[[target]], p = .7, list = FALSE, times = 1)
trainData <- df_norm[trainIndex, ]
testData <- df_norm[-trainIndex, ]

# Treinando o modelo
set.seed(123)
model <- randomForest(as.formula(paste(target, paste(vars, collapse = "+"), sep = "~")), data = trainData)

# Fazendo a previsão
pred <- predict(model, newdata = testData)

# Avaliando a precisão do modelo
accuracy <- postResample(pred, testData[[target]])
print(paste0("Precisão do modelo: ", round(accuracy, 4) * 100, "%")[2])


### TABELA DE PREVISÕES ###

# Criando uma tabela com previsões e valores reais
results <- data.frame(Predicted = pred, Actual = testData[[target]])

# Adicionando uma coluna com a diferença entre previsão e valor real
results$Difference <- results$Predicted - results$Actual

# Adicionando uma coluna com o erro percentual
results$ErrorPercent <- abs(results$Difference / results$Actual) * 100

# Mostrando resultados
head(results)



### GRÁFICO COM A PRECISÃO DO MODELO ###

# criando um data frame com os valores reais e previstos
resultados <- data.frame(Real = testData$MeanEnergyConsumptionKWH100km, Previsto = pred)

# criando o gráfico de dispersão
ggplot(data = resultados, aes(x = Real, y = Previsto)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Valor Real", y = "Valor Previsto", title = "Gráfico de Precisão do Modelo")

