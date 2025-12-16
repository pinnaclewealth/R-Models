install.packages("kernlab")
install.packages("kknn")
library(kernlab)
library(kknn)

df <- read.table("C:\\Users\\Anand\\Documents\\Georgia Tech Master's in Analytics\\Fall 2025\\ISYE 6501 - Intro to Analytics Modeling\\Week1\\data 2.2\\credit_card_data.txt", header = FALSE)
head(df)
tail(df)
summary(df)

# see ksvm details
?ksvm

# predictors set all columns except for 11
x <- df[,-11]

# target column set
y <- as.factor(df[,11])

# call ksvm (Vanilladot in  a simple linear kernel)
model <- ksvm(V11~., data = df,type="C-svc",kernel="vanilladot",
              C=100,scaled=TRUE)

# display model
model

# calculate the weights of the hyperplane v1..vm to use f[x] = w*x + b
a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
# print the coefficients
a

# calculate a0 for b or intercept
a0 <- model@b
a0

# compute training accuracy
pred <- predict(model,x)
accuracy1 <- mean(pred == y)*100 # source: https://stackoverflow.com/questions/56223887/how-to-calculate-accuracy-based-on-predictor-and-true-values-in-r
accuracy1

# adjust c values method test 10^-5 up to 10^5
c_vals <- 10^(-5:5)
results <- data.frame(C= c_vals, Accuracy = NA) # source: https://www.youtube.com/watch?v=s3GHpOvysMI&t=564s

# for loop to go through various c values
# source: https://r4ds.had.co.nz/iteration.html

for (i in seq_along(c_vals)) {
  model_c <- ksvm(V11~., data = df,type="C-svc",kernel="vanilladot",
                     C= c_vals[i],scaled=TRUE)
  # update the accuracy of the model prediction for each value of c in results data frame
  results$Accuracy[i] <- mean(predict(model_c,x) ==y)*100
}

print(results)


# rbfdot kernel method
results_rbf <- data.frame(C = c_vals, Accuracy = NA) 

for (i in seq_along(c_vals)) {
  model_rbf <- ksvm(V11~., data=df,type="C-svc",kernel="rbfdot",
                kpar=list(sigma = 0.1),C= c_vals[i],scaled=TRUE)
  # update the accuracy of the model prediction for each value of c in results data frame
  results_rbf$Accuracy[i] <- mean(predict(model_rbf,x) == y)*100
}

print(results_rbf)


# kkNN method
?kknn

# set i 
i <- 1
set.seed(123)

# kknn version 1 manual
model_kknn = kknn(V11~., df[-i,], df[i,], k = 10, distance = 2, kernel = "optimal", scale = TRUE)
fitted.values(model_kknn)
df$V11 <- as.factor(df$V11)


# create list of k values and list of accuracy values
k_vals <- c(1:30)
results_kknn <- data.frame(k = k_vals, Accuracy = NA)

# loop through kknn model for each k value
for (j in seq_along(k_vals)) {
  
  # create vector of i rows from df
  preds <- rep(NA, nrow(df))
  
  # loop through kknn model for each i while leaving out the test column to be i itself 
  for (i in 1:nrow(df)) {
    model_kknn = kknn(V11~., df[-i,], df[i,], k = k_vals[j], distance = 2, kernel = "optimal", scale = TRUE)
  
  # check each prediction as character instead of factor for each fitted class for i left out row
  preds[i] <- as.character(fitted(model_kknn))
  }
  # average of accuracy of each k values predictions across each i
  results_kknn$Accuracy[j] <- mean(preds == df$V11) *100 
}
print(results_kknn)