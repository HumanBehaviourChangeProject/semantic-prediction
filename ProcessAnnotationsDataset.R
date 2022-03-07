setwd("data")

r = getOption("repos")
r["CRAN"] = "https://ftp.fau.de/cran/"
options(repos = r)
#

# Required libraries
#debug(utils:::unpackPkgZip)
if (!require('xlsx',quietly = T)) install.packages('xlsx'); library('xlsx') #install.packages('xlsx', INSTALL_opts='--no-multiarch'); library('xlsx')
if (!require('tidyverse',quietly = T)) install.packages('tidyverse'); library(tidyverse)
if (!require('lme4',quietly=T)) install.packages('lme4') ; library(lme4)
if (!require('jtools',quietly = T)) install.packages('jtools'); library(jtools)
# The next two are required by jtools
if (!require('ggstance', quietly=T)) install.packages('ggstance'); library(ggstance)
if (!require('broom.mixed',quietly=T)) install.packages('broom.mixed'); library(broom.mixed)
#if (!require("boot",quietly=T)) install.packages("boot"); library(boot) # For bootstrapping p values for the model 
if (!require("vioplot")) install.packages("vioplot"); library(vioplot)
if (!require("merTools")) install.packages("merTools"); library(merTools)
if (!require("leaps")) install.packages("leaps"); library(leaps)


# Load the data 

df.attrs <- read.xlsx("cleaned_dataset_13Feb2022_notes_removed_control-2.xlsx", 
								sheetIndex = 1)

## Various cleaning transformations

df.clean <- df.attrs
#df.clean$remove.paper <- unlist(lapply(df.clean$remove.paper, function(x) if (is.na(x)) 0 else x))
#df.clean$manually.added.outcome.value <- as.numeric(unlist(lapply(df.clean$manually.added.outcome.value, function(x) if (is.na(x)) 0 else x)))
#df.clean$Outcome.value <- as.numeric(unlist(lapply(df.clean$Outcome.value, function(x) if (is.na(x) | x=='-') 0 else x)))

#df.clean <- df.clean[! (df.clean$remove.paper == '1') , ]
#df.clean$Outcome.value <- unlist(mapply(df.clean$Outcome.value,df.clean$manually.added.outcome.value, 
#																				FUN=function(x,y) if (y > 0) y else x ) ) 

df.clean$NA. <- NULL
df.clean$NA..1 <- NULL
#df.clean$manually.added.outcome.value=NULL
#df.clean$Will.be.fixable.in.EPPI...will.need.to.update.the.JSON.file=NULL
#df.clean$will.need.to.be.fixed.via.supplementary.file.merge=NULL
df.clean$Manually.added.follow.up.duration.units <- NULL
df.clean$Manually.added.follow.up.duration.value <- NULL
df.clean$document=NULL
df.clean$arm=NULL
df.clean$Abstinence.type <- NULL
df.clean$Combined.follow.up <- as.numeric(df.clean$Combined.follow.up)
df.clean$Mean.age <- as.numeric(df.clean$Mean.age)
df.clean$Proportion.identifying.as.female.gender <- as.numeric(df.clean$Proportion.identifying.as.female.gender)
df.clean$Mean.number.of.times.tobacco.used <- as.numeric(df.clean$Mean.number.of.times.tobacco.used)	
df.clean$Individual.level.analysed <- as.numeric(df.clean$Individual.level.analysed)
df.clean[is.na(df.clean)]=0
names(df.clean)[names(df.clean) == 'NEW.Outcome.value'] <- 'Outcome.value'

colnames.vars <- colnames(df.clean)[3:54]
colnames.vars <- colnames.vars[!colnames.vars %in% c("Country.of.intervention")]

# A mixed effects regression with study as a clustering variable 
reg.form = as.formula(paste("Outcome.value ~ ",
														paste(colnames.vars,collapse=" + "),
														" + (1 | document_id) "))

model_mixed = lmer(reg.form, 
									 data = df.clean,na.action = na.exclude)
summary(model_mixed)
confint(model_mixed)

print(model_mixed, correlation=TRUE)

par(mar=c(11,4,4,4))
barplot(fixef(model_mixed),las=2,cex.names = 0.5)

summ(model_mixed)
plot_summs(model_mixed)

plot(fitted(model_mixed), resid(model_mixed, type = "pearson"))# this will create the plot
abline(0,0, col="red")

qqnorm(resid(model_mixed)) 
qqline(resid(model_mixed), col = "red") # add a perfect fit line

res.predict = predict(model_mixed)

plot(density(df.clean$Outcome.value),ylim=c(0,0.1),main="Outcome value vs. Prediction")
lines(density(res.predict),col='blue')
legend("topright",legend=c("Outcomes","Predicted outcomes"),pch=15,col=c("black","blue"))
			 
plot(df.clean$Outcome.value,res.predict,xlab="Outcome value",
		 ylab="Predicted outcome value",main="Outcome vs. Prediction",pch=16)

plot(df.clean$Outcome.value,mapply(df.clean$Outcome.value,res.predict,FUN=function(x,y) {abs(x-y)}),
		 main="Absolute Error of Prediction",xlab="Outcome value", ylab="Error",pch=16)

# What is the error of the prediction ? 

RMSE.merMod(model_mixed, scale = FALSE)
# This gives => 5.27218   UPDATE=5.244

mean((df.clean$Outcome.value - res.predict)^2)
# This gives => 27.79589

sqrt(mean((df.clean$Outcome.value - res.predict)^2))
# This gives => 5.272181, i.e. corresponding to that given by the merMod method above. 

plot(density(df.clean$Outcome.value - res.predict),main="Error Distribution")


### Now we use a train/test split to evaluate the model 
### split based on document_id 

res.validations <- unlist(lapply(1:200, function(x) {
	docs_intrain <- sample(unique(df.clean$document_id), size = length(unique(df.clean$document_id))*0.8)
	train <- df.clean[df.clean$document_id %in% docs_intrain,]
	test <- df.clean[!df.clean$document_id %in% docs_intrain,]

	ov_lmm1 <- lmer(reg.form, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	sqrt(mean((test$Outcome.value - ov_lmm1_prd)^2))
}))

print(paste("Mean for documents is", mean(res.validations)))
#sd(res.validations)
#boxplot(res.validations,main="RMSE for 80:20 split on document ID",pch=16)

### split based on arm_id 

res.validations.arms <- unlist(lapply(1:200, function(x) {
	arms_intrain <- sample(unique(df.clean$arm_id), size = length(unique(df.clean$arm_id))*0.8)
	train <- df.clean[df.clean$arm_id %in% arms_intrain,]
	test <- df.clean[!df.clean$arm_id %in% arms_intrain,]
	
	ov_lmm1 <- lmer(reg.form, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	sqrt(mean((test$Outcome.value - ov_lmm1_prd)^2))
}))

print(paste("Mean for arms is ",mean(res.validations.arms)))
#sd(res.validations)
boxplot(list("ARMS"=res.validations.arms,"DOCUMENTS"=res.validations),main="RMSE for 80:20 split",pch=16)


### How does it look with different variables and combinations of variables
# empirical evaluation of predictive power of different hierarchical levels of data annotation in the dataset (e.g. pharmacological support/buproprion; healthcare worker/doctor), which lead to the best predictive performance

# REDO THIS ANALYSIS WITH THE HIERARCHICAL APPROACH FROM THE ONTOLOGY


# First try a data-driven statistical approach to select the best predictors from the dataset 
lm1 <- lm(df.clean,formula=Outcome.value ~ control + Biochemical.verification + CBT + Motivational.Interviewing + 
						brief.advise + X1.1.Goal.setting + X1.2.Problem.solving + 
						X1.4.Action.planning + X2.2.Feedback.on.behaviour + X2.3.Self.monitoring.of.behavior + 
						X3.1.Social.support + X4.1.Instruction.on.how.to.perform.the.behavior + 
						X4.5..Advise.to.change.behavior + X5.1.Information.about.health.consequences + 
						X5.3.Information.about.social.and.environmental.consequences + 
						X11.1.Pharmacological.support + X11.2.Reduce.negative.emotions + 
						Group.based + Face.to.face + phone + Website...Computer.Program...App + 
						Patch + Somatic + gum + e_cigarette + inhaler + lozenge + 
						nasal_spray + placebo + nrt + Pill + bupropion + varenicline + 
						Printed.material + Digital.content.type + text.messaging + 
						Health.Professional + doctor + nurse + pychologist + aggregate.patient.role + 
						Mean.age + Proportion.identifying.as.female.gender + Mean.number.of.times.tobacco.used + 
						healthcare.facility + Pharmaceutical.company.funding + Pharmaceutical.company.competing.interest + 
						Individual.level.analysed + Combined.follow.up + Abstinence..Continuous. + 
						Abstinence..Point.Prevalence.)
summary(lm1)

Best_Subset <-
	regsubsets(Outcome.value ~ control + Biochemical.verification + CBT + Motivational.Interviewing + 
						 	brief.advise + X1.1.Goal.setting + X1.2.Problem.solving + 
						 	X1.4.Action.planning + X2.2.Feedback.on.behaviour + X2.3.Self.monitoring.of.behavior + 
						 	X3.1.Social.support + X4.1.Instruction.on.how.to.perform.the.behavior + 
						 	X4.5..Advise.to.change.behavior + X5.1.Information.about.health.consequences + 
						 	X5.3.Information.about.social.and.environmental.consequences + 
						 	X11.1.Pharmacological.support + X11.2.Reduce.negative.emotions + 
						 	Group.based + Face.to.face + phone + Website...Computer.Program...App + 
						 	Patch + Somatic + gum + e_cigarette + inhaler + lozenge + 
						 	nasal_spray + placebo + nrt + Pill + bupropion + varenicline + 
						 	Printed.material + Digital.content.type + text.messaging + 
						 	Health.Professional + doctor + nurse + pychologist + aggregate.patient.role + 
						 	Mean.age + Proportion.identifying.as.female.gender + Mean.number.of.times.tobacco.used + 
						 	healthcare.facility + Pharmaceutical.company.funding + Pharmaceutical.company.competing.interest + 
						 	Individual.level.analysed + Combined.follow.up + Abstinence..Continuous. + 
						 	Abstinence..Point.Prevalence.,
						 data =df.clean,
						 nbest = 1,      # 1 best model for each number of predictors
						 nvmax = 20,    # NULL for no limit on number of variables
						 force.in = NULL, force.out = NULL,
						 method = "backward") # exhaustive

summary_best_subset <- summary(Best_Subset)

as.data.frame(summary_best_subset$outmat)

# Number of predictors: 
which.max(summary_best_subset$adjr2)
# What are the best predictors: (indicated by TRUE) 
summary_best_subset$which[20,]

# Run the regression model again with only those best predictors 
best.model <- lm(Outcome.value ~ Combined.follow.up + 
								 	Biochemical.verification + 
								 	Motivational.Interviewing + 
								 	brief.advise + 
								 	X1.1.Goal.setting + 
								 	X2.3.Self.monitoring.of.behavior +
								 	X3.1.Social.support +
								 	X11.2.Reduce.negative.emotions +
								 	Group.based +
								 	Face.to.face +
								 	phone +
								 	gum +
								 	varenicline +
								 	Digital.content.type +
								 	text.messaging +
								 	doctor +
								 	Mean.age +
								 	Proportion.identifying.as.female.gender +
								 	healthcare.facility +
								 	Pharmaceutical.company.funding, data = df.clean	)
summary(best.model)

reg.form.2 <- "Outcome.value ~ Combined.follow.up + 
											 	Biochemical.verification + 
											 	Motivational.Interviewing + 
											 	brief.advise + 
											 	X1.1.Goal.setting + 
											 	X2.3.Self.monitoring.of.behavior +
											 	X3.1.Social.support +
											 	X11.2.Reduce.negative.emotions +
											 	Group.based +
											 	Face.to.face +
											 	phone +
											 	gum +
											 	varenicline +
											 	Digital.content.type +
											 	text.messaging +
											 	doctor +
											 	Mean.age +
											 	Proportion.identifying.as.female.gender +
											 	healthcare.facility +
											 	Pharmaceutical.company.funding + (1 | document_id)"
best.model.lmer = lmer(Outcome.value ~ Combined.follow.up + 
											 	Biochemical.verification + 
											 	Motivational.Interviewing + 
											 	brief.advise + 
											 	X1.1.Goal.setting + 
											 	X2.3.Self.monitoring.of.behavior +
											 	X3.1.Social.support +
											 	X11.2.Reduce.negative.emotions +
											 	Group.based +
											 	Face.to.face +
											 	phone +
											 	gum +
											 	varenicline +
											 	Digital.content.type +
											 	text.messaging +
											 	doctor +
											 	Mean.age +
											 	Proportion.identifying.as.female.gender +
											 	healthcare.facility +
											 	Pharmaceutical.company.funding + (1 | document_id) , 
											 data=df.clean)
summary(best.model.lmer)
RMSE.merMod(best.model.lmer, scale = FALSE)
# 5.492539

res.validations <- unlist(lapply(1:100, function(x) {
	docs_intrain <- sample(unique(df.clean$document_id), size = length(unique(df.clean$document_id))*0.8)
	train <- df.clean[df.clean$document_id %in% docs_intrain,]
	test <- df.clean[!df.clean$document_id %in% docs_intrain,]
	
	ov_lmm1 <- lmer(reg.form.2, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	sqrt(mean((test$Outcome.value - ov_lmm1_prd)^2))
}))

print(paste("Mean for documents is", mean(res.validations)))
#sd(res.validations)
#boxplot(res.validations,main="RMSE for 80:20 split on document ID",pch=16)


### split based on arm_id 

res.validations.arms <- unlist(lapply(1:100, function(x) {
	arms_intrain <- sample(unique(df.clean$arm_id), size = length(unique(df.clean$arm_id))*0.8)
	train <- df.clean[df.clean$arm_id %in% arms_intrain,]
	test <- df.clean[!df.clean$arm_id %in% arms_intrain,]
	
	ov_lmm1 <- lmer(reg.form.2, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	sqrt(mean((test$Outcome.value - ov_lmm1_prd)^2))
}))

print(paste("Mean for arms is ",mean(res.validations.arms)))
#sd(res.validations)
boxplot(list("ARMS"=res.validations.arms,"DOCUMENTS"=res.validations),main="RMSE for 80:20 split main predictors",pch=16)

summ(best.model.lmer)
plot_summs(best.model.lmer)


# Analysis of errors: 

errors.arms <- unlist(lapply(1:200, function(x) {
	arms_intrain <- sample(unique(df.clean$arm_id), size = length(unique(df.clean$arm_id))*0.8)
	train <- df.clean[df.clean$arm_id %in% arms_intrain,]
	test <- df.clean[!df.clean$arm_id %in% arms_intrain,]
	
	ov_lmm1 <- lmer(reg.form, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	test$Outcome.value - ov_lmm1_prd
}))

errors.docs <- unlist(lapply(1:200, function(x) {
	docs_intrain <- sample(unique(df.clean$document_id), size = length(unique(df.clean$document_id))*0.8)
	train <- df.clean[df.clean$document_id %in% docs_intrain,]
	test <- df.clean[!df.clean$document_id %in% docs_intrain,]
	
	ov_lmm1 <- lmer(reg.form, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	test$Outcome.value - ov_lmm1_prd
}))


plot(density(errors.arms),main="Errors")

fileConn<-file("output.txt")
writeLines(errors.arms, fileConn)
close(fileConn)

lines(density(errors.docs),col='blue')
legend("topright",legend=c("Arms","Docs"),pch=15,col=c("black","blue"))


mean.abserr.arms <- unlist(lapply(1:200, function(x) {
	arms_intrain <- sample(unique(df.clean$arm_id), size = length(unique(df.clean$arm_id))*0.8)
	train <- df.clean[df.clean$arm_id %in% arms_intrain,]
	test <- df.clean[!df.clean$arm_id %in% arms_intrain,]
	
	ov_lmm1 <- lmer(reg.form, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	mean(abs(test$Outcome.value - ov_lmm1_prd))
}))

mean.abserr.docs <- unlist(lapply(1:200, function(x) {
	docs_intrain <- sample(unique(df.clean$document_id), size = length(unique(df.clean$document_id))*0.8)
	train <- df.clean[df.clean$document_id %in% docs_intrain,]
	test <- df.clean[!df.clean$document_id %in% docs_intrain,]
	
	ov_lmm1 <- lmer(reg.form, data = train)
	ov_lmm1_prd <- predict(ov_lmm1, newdata = test, allow.new.levels=TRUE)
	mean(abs(test$Outcome.value - ov_lmm1_prd))
}))

boxplot(list("ARMS"=mean.abserr.arms,"DOCUMENTS"=mean.abserr.docs),main="MAE for 80:20 split",pch=16)


