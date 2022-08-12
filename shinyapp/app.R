setwd("/Users/hastingj/Work/Python/semantic-prediction/")


if (!require('xlsx',quietly = T)) install.packages('xlsx'); library('xlsx')
#if (!require('tidyverse',quietly = T)) install.packages('tidyverse'); library(tidyverse)
if (!require('lme4',quietly=T)) install.packages('lme4') ; library(lme4)
if (!require('jtools',quietly = T)) install.packages('jtools'); library(jtools)
# The next two are required by jtools
if (!require('ggstance', quietly=T)) install.packages('ggstance'); library(ggstance)
if (!require('broom.mixed',quietly=T)) install.packages('broom.mixed'); library(broom.mixed)
#if (!require("boot",quietly=T)) install.packages("boot"); library(boot) # For bootstrapping p values for the model 
if (!require("vioplot")) install.packages("vioplot"); library(vioplot)
if (!require("merTools")) install.packages("merTools"); library(merTools)
if (!require("leaps")) install.packages("leaps"); library(leaps)
if (!require("shiny")) install.packages("shiny"); library(shiny)
if (!require("data.table")) install.packages("data.table"); library(data.table)

if (!require("plotrix")) install.packages("plotrix"); library(plotrix)
if (!require("factoextra")) install.packages("factoextra"); library(plotrix)


# LOAD DATA etc. 

df.attrs <- read.xlsx("data/cleaned_dataset_13Feb2022_notes_removed_control-2.xlsx", 
											sheetIndex = 1)
df.clean <- df.attrs

## Various cleaning transformations

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


binary.var.cols <- names(df.clean)[c(4:43,48:50,53,54)]
num.var.cols <- names(df.clean)[c(44:46,51,52)]

# Set up semantic attributes
intervention <- list("CBT"="CBT",
										 "Motivational interviewing"="Motivational.Interviewing",
										 "Brief advice" = "brief.advise",                                                
										 "Goal setting"="X1.1.Goal.setting",                                      
										 "Problem solving" = "X1.2.Problem.solving",                                        
										 "Action planning" = "X1.4.Action.planning",                                        
										 "Feedback on behaviour" = "X2.2.Feedback.on.behaviour",                               
										 "Self monitoring of behaviour" = "X2.3.Self.monitoring.of.behavior" ,                           
										 "Social support" = "X3.1.Social.support"  ,                                       
										 "Instruction on how to perform behaviour" = "X4.1.Instruction.on.how.to.perform.the.behavior",        
										 "Advise to change behaviour" = "X4.5..Advise.to.change.behavior"   ,                          
										 "Information about health consequences" = "X5.1.Information.about.health.consequences"     ,             
										 "Information about social and environmental consequences" = "X5.3.Information.about.social.and.environmental.consequences",
										 "Pharmacological support" = "X11.1.Pharmacological.support"    ,                           
										 "Reduce negative emotions" = "X11.2.Reduce.negative.emotions" )

delivery = list("Group based" = "Group.based",                                             
								"Face to face" = "Face.to.face",                                           
								"Phone" = "phone"     ,                                                  
								"Website, computer program or app" = "Website...Computer.Program...App",
								"Text messaging" = "text.messaging"   
	)

pharmacological = list ("Patch" = "Patch"  ,                                                     
												"Somatic" = "Somatic"   ,                                                  
												"Gum" = "gum"       ,                                                  
												"E-cigarette" = "e_cigarette"  ,                                               
												"Inhaler" = "inhaler"   ,                                                  
												"Lozenge" = "lozenge"   ,                                                  
												"Nasal spray" = "nasal_spray"    ,                                             
												"Placebo" = "placebo"   ,                                                  
												"NRT" = "nrt"    ,                                                     
												"Pill" = "Pill"      ,                                                  
												"Buproprion" = "bupropion"     ,                                              
												"Varenicline" = "varenicline"  )

source = list("Health professional" = "Health.Professional",                                         
						  "Doctor" = "doctor" ,                                                     
							"Nurse" = "nurse" ,                                                      
							"Psychologist" = "pychologist" )

pharma = list("Pharmaceutical company funding" = "Pharmaceutical.company.funding"  ,
							"Pharmaceutical company competing interest" ="Pharmaceutical.company.competing.interest")

outcome = list("Continuous abstinence" = "Abstinence..Continuous." ,                                    
							 "Point prevalence abstinence" = "Abstinence..Point.Prevalence." )


#### Set up interface to prediction system 
library(reticulate)
use_python("/Library/Frameworks/Python.framework/Versions/3.8/bin/python3")

os <- import("os")
os$chdir("/Users/hastingj/Work/Python/semantic-prediction")
source_python('rulenn/ui_interface.py')
initialise_model()


# Define UI for app ----
ui <- fluidPage(
	
	# App title ----
	titlePanel("HBCP predictions (prototype): Smoking Cessation"),
	
	# Sidebar layout with input and output definitions ----
	sidebarLayout(
		sidebarPanel(
		
			checkboxGroupInput(
				"intervention",
				"Intervention",
				choices = intervention
			),
			
			checkboxGroupInput(
				"delivery",
				"Delivery",
				choices = delivery
			),
			
			checkboxGroupInput(
				"source",
				"Source",
				choices = source
			),
			
			conditionalPanel(
				condition = "input.intervention.indexOf('X11.1.Pharmacological.support') > -1", 
				radioButtons(
						"pharmacological",
						"Pharmacological",
						choices= pharmacological,
						selected="placebo"
					)
			),
			
			
		),
		
		# Main panel for displaying outputs ----
		mainPanel(
			
			fluidRow(
				
				column(4,
							 
							 	sliderInput(
							 		"meanAge",
							 		"Mean age",
							 		min(df.clean$Mean.age),
							 		max(df.clean$Mean.age),
							 		mean(df.clean$Mean.age)
							 	),
							 	
							 	sliderInput(
							 		"proportionFemale",
							 		"Proportion female",
							 		0,
							 		100,
							 		mean(df.clean$Proportion.identifying.as.female.gender)
							 	)
							 
							 
				),
				column(4,
							 
							 	sliderInput(
							 		"tobaccoUsed",
							 		"Mean number of times tobacco used",
							 		min(df.clean$Mean.number.of.times.tobacco.used),
							 		30,
							 		median(df.clean$Mean.number.of.times.tobacco.used)
							 	),
							 checkboxInput(
							 	"patientRole",
							 	"Patient role?"
							 )
							 
				),
				column(4,
							 
							 		radioButtons(
							 			"outcome",
							 			"Outcome",
							 			choices = outcome
							 		),
							 		
							 		checkboxInput(
							 			"Biochemical.verification",
							 			"Biochemical verification"
							 		),
							 		
							 		sliderInput(
							 			"followup",
							 			"Follow up (weeks)",
							 			min(df.clean$Combined.follow.up),
							 			as.integer(max(df.clean$Combined.follow.up)),
							 			median(df.clean$Combined.follow.up)
							 		),
							  
				)
			),
			
			column(12,
				plotOutput(outputId = "predPlot"),
			),
			
			column(12,
				fluidRow(htmlOutput("text")),
			),
			
			column(12,
				plotOutput(outputId = "pcaPlot")
			)
		)
	)
)


# Define server logic required  ----
server <- function(input, output) {
	
#	output$summsPlot <- renderPlot({
#		plot_summs(model_mixed) #
#	})
	
	preparePrediction <- function(input) {
		test <- df.clean[1,]
		
		# Baseline
		test['document_id'] = 437572
		test[['arm_id']]=0
		test[['control']]=0
		lapply(binary.var.cols, function(x) test[[x]]<<- 0)
		
		test[['Mean.age']]=input$meanAge
		test[['Proportion.identifying.as.female.gender']]=input$proportionFemale
		test[['Mean.number.of.times.tobacco.used']]=input$tobaccoUsed
		test[['Individual.level.analysed']]=mean(df.clean$Individual.level.analysed) # 314.12
		test[['Combined.follow.up']]=input$followup
		test[['aggregate.patient.role']]=input$patientRole
		
		control <- test
		control [['control']] = 1
		
		# Set intervention attributes from inputs 
		
		lapply(binary.var.cols, function(x) {
			if (x %in% input$intervention
					| x %in% input$source
					| x %in% input$delivery)  test[[x]]<<- 1
			if ("X11.1.Pharmacological.support" %in% input$intervention) {
				if (x %in% input$pharmacological) test[[x]]<<- 1
			}
			
		})
		
		lapply(input$outcome, function(outc) {
			test[[outc]]<- 1
			control[[outc]] <- 1
		})
		
		return( list(test,control) )
	}
	
	preparePredictionRules <- function(input) {
		print(input)
	
		# Baseline

		test <- c(input$meanAge <= 12, #'Mean age (child)',
							input$meanAge <= 18, #'Mean age (adolecent)',
							input$meanAge <= 25, #'Mean age (young adult)',
							input$meanAge <= 60, #'Mean age (adult)',
							input$meanAge >= 60, #'Mean age (elderly)',
							"doctor" %in% input$source, #'doctor',
							'Patch' %in% input$pharmacological, #'Patch',
							0, #'control',
							"X2.3.Self.monitoring.of.behavior" %in% input$intervention, #'2.3 Self-monitoring of behavior',
							"Pill" %in% input$pharmacological, #'Pill',
							"X4.1.Instruction.on.how.to.perform.the.behavior" %in% input$intervention, #'4.1 Instruction on how to perform the behavior',
							"bupropion" %in% input$pharmacological, #'bupropion',
							"nurse" %in% input$source, #'nurse',
							"gum" %in% input$pharmacological, #'gum',
							"X1.1.Goal.setting" %in% input$intervention, #'1.1.Goal setting',
							"e_cigarette" %in% input$pharmacological, #'e_cigarette',
							"placebo" %in% input$pharmacological, #'placebo',
							"Website...Computer.Program...App" %in% input$delivery, #'Website / Computer Program / App',
							"nrt" %in% input$pharmacological, #'nrt',
							input$tobaccoUsed <=5, #'Mean number of times tobacco used (<= 5)',
							input$tobaccoUsed <=10, #'Mean number of times tobacco used (<= 10)',
							input$tobaccoUsed <=15, #'Mean number of times tobacco used (<= 15)',
							input$tobaccoUsed <=20, #'Mean number of times tobacco used (<= 20)',
							input$tobaccoUsed <=25, #'Mean number of times tobacco used (<= 25)',
							input$tobaccoUsed >=26, #'Mean number of times tobacco used (> 26)',
							0, #'healthcare facility',  ? not in interface yet ? 
							"Pharmaceutical.company.competing.interest" %in% input$pharma, #'Pharmaceutical company competing interest',
							"phone" %in% input$delivery, #'phone',
							"Biochemical.verification" %in% input$outcome, #'Biochemical verification',
							"X2.2.Feedback.on.behaviour" %in% input$intervention, #'2.2 Feedback on behaviour',
							"Digital.content.type" %in% input$delivery, #'Digital content type',
							"X3.1.Social.support" %in% input$intervention, #'3.1 Social support',
							"Motivational.Interviewing" %in% input$intervention, #'Motivational Interviewing',
							input$patientRole, #'aggregate patient role',
							"Health.Professional" %in% input$source, #'Health Professional',
							"X1.2.Problem.solving" %in% input$intervention, #'1.2 Problem solving',
							"Pharmaceutical.company.funding" %in% input$pharma, #'Pharmaceutical company funding',
							"nasal_spray" %in% input$pharmacological, #'nasal_spray',
							"Abstinence..Point.Prevalence." %in% input$outcome, #'Abstinence: Point Prevalence ',
							"inhaler" %in% input$pharmacological, #'inhaler',
							"pychologist" %in% input$source, #'pychologist',
							"X11.1.Pharmacological.support" %in% input$intervention, #'11.1 Pharmacological support',
							"X11.2.Reduce.negative.emotions" %in% input$intervention, #'11.2 Reduce negative emotions',
							"Face.to.face" %in% input$delivery, #'Face to face',
							"varenicline" %in% input$pharmacological, #'varenicline',
							input$followup<=1, #'Combined follow up (one week)',
							input$followup<=4, #'Combined follow up (<= 1 month)',
							input$followup<=32, #'Combined follow up (<= 8 months)',
							input$followup<=60, #'Combined follow up (<= 15 month)',
							input$followup<=80, #'Combined follow up (<= 20 months)',
							input$followup<=112, #'Combined follow up (<= 28 months)',
							input$followup>112, #'Combined follow up (> 29 months)',
							"Group.based" %in% input$delivery, #'Group-based',
							"X4.5..Advise.to.change.behavior" %in% input$intervention, #'4.5. Advise to change behavior',
							"X5.3.Information.about.social.and.environmental.consequences" %in% input$intervention, #'5.3 Information about social and environmental consequences',
							"X1.4.Action.planning" %in% input$intervention, #'1.4 Action planning',
							"Printed.material" %in% input$delivery, #'Printed material',
							"Abstinence..Continuous." %in% input$outcome, #'Abstinence: Continuous ',
							input$proportionFemale==0, #'Proportion identifying as female gender (None)',
							input$proportionFemale<=15, #'Proportion identifying as female gender (<= 15)',
							input$proportionFemale<=35, #'Proportion identifying as female gender (<= 35)',
							input$proportionFemale<=45, #'Proportion identifying as female gender (<= 45)',
							input$proportionFemale<=60, #'Proportion identifying as female gender (<= 60)',
							input$proportionFemale<=99, #'Proportion identifying as female gender (<= 99)',
							input$proportionFemale==100, #'Proportion identifying as female gender (All)',
							"CBT" %in% input$intervention, #'CBT',
							1,#input$Individual.level.analysed<=100, #'Individual-level analysed (~100)',
							1,#input$Individual.level.analysed<=500, #'Individual-level analysed (~500)',
							0,#input$Individual.level.analysed<=1000, #'Individual-level analysed (~1000)',
							0,#input$Individual.level.analysed<=3000, #'Individual-level analysed (~3000)',
							0,#input$Individual.level.analysed<=15000, #'Individual-level analysed (~15000)',
							"lozenge" %in% input$pharmacological, #'lozenge',
							"Somatic" %in% input$pharmacological, #'Somatic',
							"brief.advise" %in% input$intervention, #'brief advise',
							"text.messaging" %in% input$delivery, #'text messaging',
							"X5.1.Information.about.health.consequences" %in% input$delivery) #'5.1 Information about health consequences']
		
		control <- test
		control[8] <- 1 # 'control'
		control[c(6:19,
							 28:33,
							 35,36,
							 38,
							 40:45,
							 53:57,
							 66,
							 72:76)] <- 0
		
		return( list(test,control) )
	}
	
	output$text <- renderText(
		{ # The dataset we are expecting is 
			
			vals = preparePredictionRules(input)
			
			test  = vals[[1]]
			print(test)
			
			control = vals[[2]]
			print(control)
			
			#pcacols <- c(3:46,48:54)
			#df.datavars <- df.clean[,pcacols]
			#rownames(df.datavars) <- paste(df.attrs$document,'-',df.attrs$arm)
			
			#pca <- prcomp(df.datavars,scale=T)
			
			#testpoint <- predict(pca,newdata = test)
			#ctrlpoint <- predict(pca,newdata = control)
			
			#testdists <- as.numeric((pca$x[,1]-testpoint[1])^2
			#												+(pca$x[,2]-testpoint[2])^2)
			#closesttests <- paste(rownames(pca$x)[order(testdists)][1:3],collapse='</li><li>')
			#ctrldists <- as.numeric((pca$x[,1]-ctrlpoint[1])^2
			#												+(pca$x[,2]-ctrlpoint[2])^2)
			#closestctrls <- paste(rownames(pca$x)[order(ctrldists)][1:3],collapse='</li><li>')
			
			#print(testdists)
			
			#paste("<b>Closest to baseline: </b><br/><ul><li>",closestctrls,"</li></ul><br/><b>Closest to intervention: </b><br/><ul><li>",closesttests,"</li></ul>")  # Todo: Which is the most likely intervention to succeed for the given population/setting/etc? 
			
			predtest <- get_prediction(test)
			predctrl <- get_prediction(control)
			
			paste("Values: test ",predtest[1], ", control ",predctrl[1], '<br/><br/>',
			"Test: ",paste(predtest[2],sep = '<br/>'),'<br/><br/>',
			"Control: ",paste(predctrl[2],sep='<br/>') )
			
		}
	)
	
	output$predPlot <- renderPlot({
		#print(input$intervention)
		
		vals = preparePrediction(input)
		test  = vals[[1]]
		control = vals[[2]]
		
		pred <- predictInterval(model_mixed,
														newdata=rbind.data.frame(control,test),
														n.sims = 599,
														level=0.5)
		
		plotCI(x = c(1,2),               # plotrix plot with confidence intervals
					 y = pred$fit,
					 li = pred$lwr,
					 ui = pred$upr,
					 col=c('blue','red'),
					 scol='grey',
					 pch=16,
					 ylab="Predicted outcome value (% cessation)",
					 ylim=c(0,50),
					 xaxt='n',
					 xlim=c(0,3),
					 xlab=NA,
					 main="Prediction")
		axis(side=1, at=c(1,2), labels = c("Baseline","Intervention"))
	})
	
	
	output$pcaPlot <- renderPlot({
		# Most similar paper ? Could be determined by a similarity metric to the annotation dataset ? 
		pcacols <- c(3:46,48:54)
		df.datavars <- df.clean[,pcacols]
		rownames(df.datavars) <- paste(df.attrs$document,'-',df.attrs$arm)
		
		pca <- prcomp(df.datavars,scale=T)
		
		vals = preparePrediction(input)
		test  = vals[[1]]
		control = vals[[2]]
		
		point <- predict(pca,newdata = test)
		ctrlp <- predict(pca,newdata = control)
		
		fviz_pca_ind(pca, 
								 geom.ind = "point", 
								 pointshape = 21, 
								 pointsize = 2, 
								 fill.ind = factor(df.datavars$control), 
								 col.ind = "black", 
								 palette = c("lightgrey", "darkgrey"),#"jco", 
								 addEllipses = FALSE,
								 label = "var",
								 col.var = "black",
								 repel = TRUE,
								 legend.title = "Control") +
			ggtitle("", subtitle = waiver()) +
			annotate("point", x=point[1], y=point[2], colour="red",size = 3) + 
			annotate("point", x=ctrlp[1], y=ctrlp[2], colour="blue",size = 3)
		
	})
	
}

options(shiny.port = 5001)
shinyApp(ui = ui, server = server)

