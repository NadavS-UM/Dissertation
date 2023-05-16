
### 1. Calculate 95% CIs for WER by bootstrapping

# Set seed for reproducibility
set.seed(12345)        

# Hebrew speakers
hebrew<-c(2.00,3.18,1.33,1.76,1.08)
range(hebrew)
mean(hebrew)

n=10000
sampling<-vector(length=n)
for (i in 1:n) {
  sampling[i]<-mean(sample(hebrew, size=length(hebrew), replace=TRUE))
}

mean(sampling)
sd(sampling)
alpha = 0.05
quantile(sampling, probs = c(alpha/2,1-alpha/2))

# English speakers
english<-c(0.61, 1.59, 3.96,1.29,0.32)
range(english)
mean(english)

n=10000
sampling<-vector(length=n)
for (i in 1:n) {
  sampling[i]<-mean(sample(english, size=length(english), replace=TRUE))
}

mean(sampling)
sd(sampling)
alpha = 0.05
quantile(sampling, probs = c(alpha/2,1-alpha/2))

### 2. Statistical tests

# Create the data frame
language <- c("English","English","English","English","English","English",
              "English","English","English","English","English","English",
              "Hebrew", "Hebrew", "Hebrew", "Hebrew", "Hebrew", "Hebrew",
              "Hebrew", "Hebrew", "Hebrew", "Hebrew", "Hebrew", "Hebrew")
error <- c(9,21,4,6,9,7,
           10,25,40,24,5,3,
           8,7,8,14,24,12,
           8,35,26,14,8,0)

df <- data.frame(language, error)
df$language = factor(df$language, levels=unique(df$language))

#  Check the data frame
df
str(df)
summary(df)

# Poisson regression

model = glm(error ~ language,
              data=df,
              family="poisson")

model.null = glm(error ~ 1,
                data=df,
                family="poisson")

anova(model, model.null)

library(car)
Anova(model,
      type="II",
      test="LR")

# G^2(1)=0.003, p=.95

