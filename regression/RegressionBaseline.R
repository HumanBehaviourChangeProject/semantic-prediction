setwd("/Users/hastingj/Work/Python/semantic-prediction/regression")

# Required libraries

if (!require('xlsx',quietly = T)) install.packages('xlsx'); library('xlsx')
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

if (!require("plotrix")) install.packages("plotrix"); library(plotrix)
if (!require("factoextra")) install.packages("factoextra"); library(plotrix)


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

# Clean data: these attributes should be imputed (using the mean value, check median as well) 
df.clean$Mean.age[is.na(df.clean$Mean.age)] <- mean(df.clean$Mean.age,na.rm=T)
df.clean$Proportion.identifying.as.female.gender[is.na(df.clean$Proportion.identifying.as.female.gender)] <- mean(df.clean$Proportion.identifying.as.female.gender,na.rm=T)
df.clean$Mean.number.of.times.tobacco.used[is.na(df.clean$Mean.number.of.times.tobacco.used)] <- mean(df.clean$Mean.number.of.times.tobacco.used,na.rm=T)
# For the rest of the attributes, replace NAs with 0
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
#confint(model_mixed)

#print(model_mixed, correlation=TRUE)

par(mar=c(11,4,4,4))
barplot(fixef(model_mixed),las=2,cex.names = 0.5)

summ(model_mixed)
plot_summs(model_mixed)

#plot(fitted(model_mixed), resid(model_mixed, type = "pearson"))# this will create the plot
#abline(0,0, col="red")

#qqnorm(resid(model_mixed)) 
#qqline(resid(model_mixed), col = "red") # add a perfect fit line

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
# This gives => 5.27218   UPDATE=5.244  UPDATE AFTER IMPUTATION: 5.229

mean((df.clean$Outcome.value - res.predict)^2)
# This gives => 27.79589

meanrmse <- sqrt(mean((df.clean$Outcome.value - res.predict)^2))
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




if (!require("BiocManager", quietly = TRUE))
	install.packages("BiocManager")

BiocManager::install("ropls")
library(ropls)

df.clean$Country.of.intervention <- NULL

# We can estimate the predictive performance of each of the variables using OPLS-DA
## ----oplsda-------------------------------------------------------------------
df.oplsda <- opls(df.clean[,1:53], df.clean[,54],
												predI = 5, orthoI = NA)


vipVn <- getVipVn(df.oplsda)
par(mar=c(10.1, 4.1,  4.1, 2.1))

barplot(vipVn[order(vipVn)],las=2,cex.names = 0.8)
abline(h=1,col='red',lty='dashed')
par(mar=c(5.1, 4.1,  4.1, 2.1))

# Now we evaluate the sensitivity of the model to outliers. 
# We could define outliers as those with the highest error in their prediction
# Data-driven approach: Remove a document, recheck MAE and RMSE for model

docs.means <- unlist(lapply(unique(df.clean$document_id), function(doc_id) {
	df.subs <- df.clean[which(df.clean$document_id != doc_id),]
	model_mixed = lmer(reg.form, 
										 data = df.subs,na.action = na.exclude)
	res.predict = predict(model_mixed)
	meanrmse <- sqrt(mean((df.subs$Outcome.value - res.predict)^2))
	
	meanrmse
}))
names(docs.means) <- unique(df.clean$document_id)

plot(density(docs.means))
docs.means[order(docs.means)]



# 
# ## ----oplsda_subset, warning=FALSE---------------------------------------------
# sacurine.oplsda <- opls(dataMatrix, genderFc,
# 												predI = 1, orthoI = NA,
# 												subset = "odd")
# 
# ## ----train--------------------------------------------------------------------
# trainVi <- getSubsetVi(sacurine.oplsda)
# table(genderFc[trainVi], fitted(sacurine.oplsda))
# 
# ## ----test---------------------------------------------------------------------
# table(genderFc[-trainVi],
# 			predict(sacurine.oplsda, dataMatrix[-trainVi, ]))
# 
# ## ----overfit, echo = FALSE----------------------------------------------------
# set.seed(123)
# obsI <- 20
# featVi <- c(2, 20, 200)
# featMaxI <- max(featVi)
# xRandMN <- matrix(runif(obsI * featMaxI), nrow = obsI)
# yRandVn <- sample(c(rep(0, obsI / 2), rep(1, obsI / 2)))
# 
# layout(matrix(1:4, nrow = 2, byrow = TRUE))
# for (featI in featVi) {
# 	randPlsi <- opls(xRandMN[, 1:featI], yRandVn,
# 									 predI = 2,
# 									 permI = ifelse(featI == featMaxI, 100, 0),
# 									 fig.pdfC = "none",
# 									 info.txtC = "none")
# 	plot(randPlsi, typeVc = "x-score",
# 			 parCexN = 1.3, parTitleL = FALSE,
# 			 parCexMetricN = 0.5)
# 	mtext(featI/obsI, font = 2, line = 2)
# 	if (featI == featMaxI)
# 		plot(randPlsi,
# 				 typeVc = "permutation",
# 				 parCexN = 1.3)
# }
# mtext(" obs./feat. ratio:",
# 			adj = 0, at = 0, font = 2,
# 			line = -2, outer = TRUE)
# 
# ## ----vip----------------------------------------------------------------------
# ageVn <- sampleMetadata[, "age"]
# 
# pvaVn <- apply(dataMatrix, 2,
# 							 function(feaVn) cor.test(ageVn, feaVn)[["p.value"]])
# 
# 
# quantVn <- qnorm(1 - pvaVn / 2)
# rmsQuantN <- sqrt(mean(quantVn^2))
# 
# opar <- par(font = 2, font.axis = 2, font.lab = 2,
# 						las = 1,
# 						mar = c(5.1, 4.6, 4.1, 2.1),
# 						lwd = 2, pch = 16)
# 
# plot(pvaVn, vipVn,
# 		 col = "red",
# 		 pch = 16,
# 		 xlab = "p-value", ylab = "VIP", xaxs = "i", yaxs = "i")
# 
# box(lwd = 2)
# 
# curve(qnorm(1 - x / 2) / rmsQuantN, 0, 1, add = TRUE, col = "red", lwd = 3)
# 
# abline(h = 1, col = "blue")
# abline(v = 0.05, col = "blue")
# 
# par(opar)


# --------------------------------------
# Test specific variables in predictions
# --------------------------------------


# Create the 'default' starting sample, no interventions, just means of data points
test <- df.clean[1,]

test[['document_id']]=0
test[['arm_id']]=0
test[['control']]=0
test[['Biochemical.verification']]=0
test[['Website...Computer.Program...App']]=0
test[['Mean.age']]=mean(df.clean$Mean.age)  # 34.46
test[['Proportion.identifying.as.female.gender']]=mean(df.clean$Proportion.identifying.as.female.gender) #35.24
test[['Mean.number.of.times.tobacco.used']]=mean(df.clean$Mean.number.of.times.tobacco.used) # 13.75
test[['Pharmaceutical.company.competing.interest']]=0
test[['Individual.level.analysed']]=mean(df.clean$Individual.level.analysed) # 314.12
test[['Combined.follow.up']]=mean(df.clean$Combined.follow.up) # 38.91

default <- test

test[['control']]=1
v.ctrl <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
test[['control']]=0
v.pred <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)

plot(x=c(1,2),y=c(v.ctrl,v.pred),ylim=c(0,30),
		 xlim=c(0,3),pch=16,xaxt='n',xlab=NA,ylab="Predicted Outcome (% cessation)")
axis(1, at=c(1,2), labels=c("Control","Intervention")) 

test <- default



# Specific intervention types (BCTs) 

interventions <- colnames(df.clean)[5:19]

res.interventions <- unlist(lapply(interventions, function(interv) {
	test[[interv]] <- 1
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	test[[interv]] <- 0
	res.int
}))

par(mar=c(10,4,2,2))
plot(x=1:(length(res.interventions)+1),
		 y=c(v.ctrl,res.interventions),ylim=c(0,30),
		 xlim=c(0,(length(res.interventions)+2)),pch=16,xaxt='n',
		 xlab=NA,ylab="Predicted Outcome (% cessation)")
axis(1, at=1:(length(res.interventions)+1), 
		 labels=c("Control",interventions),las=2,cex.axis=0.5) 
par(mar=c(5.1, 4.1, 4.1, 2.1))

test=default


# Modes of Delivery 

interventions <- colnames(df.clean)[c(20:23,36:38)]

res.interventions <- unlist(lapply(interventions, function(interv) {
	test[[interv]] <- 1
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	test[[interv]] <- 0
	res.int
}))

par(mar=c(10,4,2,2))
plot(x=1:(length(res.interventions)+1),
		 y=c(v.ctrl,res.interventions),ylim=c(0,30),
		 xlim=c(0,(length(res.interventions)+2)),pch=16,xaxt='n',
		 xlab=NA,ylab="Predicted Outcome (% cessation)")
axis(1, at=1:(length(res.interventions)+1), 
		 labels=c("Control",interventions),las=2,cex.axis=0.5) 
par(mar=c(5.1, 4.1, 4.1, 2.1))

test=default

# NRT types

interventions <- colnames(df.clean)[24:35]

res.interventions <- unlist(lapply(interventions, function(interv) {
	test[[interv]] <- 1
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	test[[interv]] <- 0
	res.int
}))

par(mar=c(10,4,2,2))
plot(x=1:(length(res.interventions)+1),
		 y=c(v.ctrl,res.interventions),ylim=c(0,30),
		 xlim=c(0,(length(res.interventions)+2)),pch=16,xaxt='n',
		 xlab=NA,ylab="Predicted Outcome (% cessation)")
axis(1, at=1:(length(res.interventions)+1), 
		 labels=c("Control",interventions),las=2,cex.axis=0.5) 
par(mar=c(5.1, 4.1, 4.1, 2.1))

test=default

# People 


interventions <- colnames(df.clean)[39:43]

res.interventions <- unlist(lapply(interventions, function(interv) {
	test[[interv]] <- 1
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	test[[interv]] <- 0
	res.int
}))

par(mar=c(10,4,2,2))
plot(x=1:(length(res.interventions)+1),
		 y=c(v.ctrl,res.interventions),ylim=c(0,30),
		 xlim=c(0,(length(res.interventions)+2)),pch=16,xaxt='n',
		 xlab=NA,ylab="Predicted Outcome (% cessation)")
axis(1, at=1:(length(res.interventions)+1), 
		 labels=c("Control",interventions),las=2,cex.axis=0.5) 
par(mar=c(5.1, 4.1, 4.1, 2.1))

test=default


# Setting and outcome type

interventions <- colnames(df.clean)[c(48,49,50,52,54)]

res.interventions <- unlist(lapply(interventions, function(interv) {
	test[[interv]] <- 1
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	test[[interv]] <- 0
	res.int
}))

par(mar=c(10,4,2,2))
plot(x=1:(length(res.interventions)+1),
		 y=c(v.ctrl,res.interventions),ylim=c(0,30),
		 xlim=c(0,(length(res.interventions)+2)),pch=16,xaxt='n',
		 xlab=NA,ylab="Predicted Outcome (% cessation)")
axis(1, at=1:(length(res.interventions)+1), 
		 labels=c("Control",interventions),las=2,cex.axis=0.5) 
par(mar=c(5.1, 4.1, 4.1, 2.1))

test=default




# 
# Numeric attributes
# 

attr.name <- "Mean.age"

plot(df.clean[[attr.name]],df.clean$Outcome.value,pch=16,xlab=attr.name,
		 col=unlist(lapply(df.clean$control,function(x) {if (x==0) {"black"} else {"grey"}})),
		 ylab="Outcome (% cessation)")


attr.values <- seq(0,100,by=5)
attr.default <- mean(df.clean[[attr.name]])

res.attrs <- unlist(lapply(attr.values, function(attrval) {
	test[[attr.name]] <- attrval
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	print(res.int)
	test[[attr.name]] <- attr.default
	res.int
}))

plot(x=attr.values,
		 	y=res.attrs,
		 pch=16, ylim=c(0,30),
		 xlab=attr.name,
		 ylab="Predicted Outcome (% cessation)")
abline(v=attr.default,col='red',lty=2)

test=default

# Proportion identifying as female

attr.name <- "Proportion.identifying.as.female.gender"

plot(df.clean[[attr.name]],df.clean$Outcome.value,pch=16,xlab=attr.name,
		 col=unlist(lapply(df.clean$control,function(x) {if (x==0) {"black"} else {"grey"}})),
		 ylab="Outcome (% cessation)")


attr.values <- seq(0,100,by=5)
attr.default <- mean(df.clean[[attr.name]])

res.attrs <- unlist(lapply(attr.values, function(attrval) {
	test[[attr.name]] <- attrval
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	print(res.int)
	test[[attr.name]] <- attr.default
	res.int
}))

plot(x=attr.values,
		 y=res.attrs,
		 pch=16, ylim=c(0,30),
		 xlab=attr.name,
		 ylab="Predicted Outcome (% cessation)")
abline(v=attr.default,col='red',lty=2)

test=default

# Mean number of times tobacco used

attr.name <- "Mean.number.of.times.tobacco.used"

plot(df.clean[[attr.name]],df.clean$Outcome.value,pch=16,xlab=attr.name,
		 col=unlist(lapply(df.clean$control,function(x) {if (x==0) {"black"} else {"grey"}})),
		 ylab="Outcome (% cessation)")

attr.values <- seq(0,50,by=5)
attr.default <- mean(df.clean[[attr.name]])

res.attrs <- unlist(lapply(attr.values, function(attrval) {
	test[[attr.name]] <- attrval
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	print(res.int)
	test[[attr.name]] <- attr.default
	res.int
}))

plot(x=attr.values,
		 y=res.attrs,
		 pch=16, ylim=c(0,30),
		 xlab=attr.name,
		 ylab="Predicted Outcome (% cessation)")
abline(v=attr.default,col='red',lty=2)

test=default


# Combined follow up 

attr.name <- "Combined.follow.up"

plot(df.clean[[attr.name]],df.clean$Outcome.value,pch=16,xlab=attr.name,
		 col=unlist(lapply(df.clean$control,function(x) {if (x==0) {"black"} else {"grey"}})),
		 ylab="Outcome (% cessation)")

attr.values <- seq(0,150,by=5)
attr.default <- mean(df.clean[[attr.name]])

res.attrs <- unlist(lapply(attr.values, function(attrval) {
	test[[attr.name]] <- attrval
	res.int <- predict(model_mixed, newdata = test, allow.new.levels=TRUE)
	print(res.int)
	test[[attr.name]] <- attr.default
	res.int
}))

plot(x=attr.values,
		 y=res.attrs,
		 pch=16, ylim=c(0,30),
		 xlab=attr.name,
		 ylab="Predicted Outcome (% cessation)")
abline(v=attr.default,col='red',lty=2)

test=default





# Look at prediction intervals rather than specific points
# Confidence intervals coming from the fixed/random effects

preds <- predictInterval(model_mixed, newdata = df.clean, n.sims = 999,ignore.fixed.terms = T)


plotCI(x = df.clean$Outcome.value,               # plotrix plot with confidence intervals
			 y = preds$fit,
			 li = preds$lwr,
			 ui = preds$upr,
			 col='black',
			 scol='grey',
			 pch=16,
			 ylab="Predicted outcome value (% cessation)",
			 xlab="Outcome value (% cessation)")

plot(density(preds$upr-preds$lwr),main="Confidence interval around predictions")
abline(v=mean(preds$upr-preds$lwr),col="red",lty=2)

# Which document ID is average? 
ranefs <- ranef(model_mixed)
plot(density(as.numeric(ranefs$document_id[,1])),main="Random effects")
abline(v=mean(ranefs$document_id[,1]),col='red',lty=2)

ranefs$document_id[which(ranefs$document_id >0),]

# An average random effects document ID: 437572 (0.188497957)
test['document_id'] = 437572

pred <- predictInterval(model_mixed,newdata=test,n.sims = 999,ignore.fixed.terms = T)

plotCI(x = c(1),               # plotrix plot with confidence intervals
			 y = pred$fit,
			 li = pred$lwr,
			 ui = pred$upr,
			 col='black',
			 scol='grey',
			 pch=16,
			 ylab="Predicted outcome value (% cessation)",
			 ylim=c(0,50),
			 xaxt='n',
			 xlab=NA)

# Most similar paper ? Could be determined by a similarity metric to the annotation dataset ? 
pcacols <- c(3:46,48:54)
df.datavars <- df.clean[,pcacols]
rownames(df.datavars) <- paste(df.attrs$document,'-',df.attrs$arm)

pca <- prcomp(df.datavars,scale=T)

point <- predict(pca,newdata = test)

fviz_pca_ind(pca, geom.ind = "point", pointshape = 21, 
						 pointsize = 2, 
						 fill.ind = factor(df.datavars$control), 
						 col.ind = "black", 
						 palette = "jco", 
						 addEllipses = FALSE,
						 label = "var",
						 col.var = "black",
						 repel = TRUE,
						 legend.title = "Control") +
	ggtitle("2D PCA-plot") +
	theme(plot.title = element_text(hjust = 0.5)) + 
	annotate("point", x=point[1], y=point[2], colour="red",size = 3)
