# The semantic domains of the ontologies.
# These map to different "sub" ontologies (or ontology modules) within
# the overarching ontology.
#
# These are specified in the direction DOMAIN: [LIST_OF_FEATURES]
features_domains = {"study_feature": [
                        "control",
                        "Pharmaceutical company funding",
                        "Pharmaceutical company competing interest",
                        "Individual-level analysed"
    # If risk of bias attributes are available, they go here as well
                    ],
                    "intervention_bct":[
                        "CBT",
                        "Motivational Interviewing",
                        "brief advise",
                        "1.1.Goal setting",
                        "1.2 Problem solving",
                        "1.4 Action planning",
                        "2.2 Feedback on behaviour",
                        "2.3 Self-monitoring of behavior",
                        "3.1 Social support",
                        "4.1 Instruction on how to perform the behavior",
                        "4.5. Advise to change behavior",
                        "5.1 Information about health consequences",
                        "5.3 Information about social and environmental consequences",
                        "11.1 Pharmacological support",
                        "11.2 Reduce negative emotions",
                        'Awareness of others BCT',
                        'Personal resources BCT',
                        'Monitoring BCT',
                        'Consequences BCT',
                        'Goal directed BCT',
                        'External stimulus BCT',
                        'Habit BCT',
                        'Mental representation BCT',
                        'Social support BCT',
                        'Environmental restructuring BCT'

                    ],
                    "mode_of_delivery": [
                        "Group-based",
                        "Face to face",
                        "phone",
                        "Website / Computer Program / App",
                        "Patch",
                        "Somatic",
                        "gum",
                        "e_cigarette",
                        "inhaler",
                        "lozenge",
                        "nasal_spray",
                        "Pill",
                        "Printed material",
                        "Digital content type",
                        "text messaging"
                    ],
                    "source": [
                        "Health Professional",
                        "doctor",
                        "nurse"	,
                        "pychologist"
                    ],
                    "population": [
                        "aggregate patient role",
                        "Mean age",
                        "Proportion identifying as female gender",
                        "Mean number of times tobacco used"
                    ],
                    "setting": [
                        "Country of intervention",
                        "healthcare facility"
                    ],
                    "outcome": [
                        "Combined follow up",
                        "Abstinence type",
                        "Abstinence: Continuous",
                        "Abstinence: Point Prevalence",
                        "Manually added follow-up duration value",
                        "Manually added follow-up duration units",
                        "Biochemical verification",
                        "NEW Outcome value"
                    ]
             }

# Hierarchical relationships between features.
#
# These are specified in the direction PARENT : [LIST_OF_CHILDREN]
features_isa = {"Somatic":["Patch",
                        "gum",
                        "inhaler",
                        "lozenge",
                        "nasal_spray",
                        "Pill"],
                "nrt": ["e_cigarette"],
                "Abstinence type": [
                        "Abstinence: Continuous",
                        "Abstinence: Point Prevalence"],
                "Health Professional": [
                        "doctor",
                        "nurse"	,
                        "pychologist"
                    ],
                "11.1 Pharmacological support": ["nrt",
                                                 "bupropion,"
                                                 "varenicline",
                                                 "placebo"],
                "Digital content type": ["Website / Computer Program / App",
                                         "text messaging"],
                "Goal directed BCT": ["1.1.Goal setting (behavior)",
                                      "1.2 Problem solving",
                                      "1.3 Goal setting (outcome)",
                                      "1.4 Action planning",
                                      "1.5 Review behavior goal(s)",
                                      "1.6 Discrepancy between current behavior and goal",
                                      "1.7 Review outcome goal(s)",
                                      "1.8 Behavioral contract",
                                      "1.9 Commitment"],
                "Monitoring BCT": ["2.1 Monitoring of behavior by others without feedback",
                                   "2.2 Feedback on behaviour",
                                   "2.3 Self-monitoring of behavior",
                                   "2.4 Self-monitoring of outcome(s) of behaviour",
                                   "2.5 Monitoring of outcome(s) of behavior without feedback",
                                   "2.6 Biofeedback",
                                   "2.7 Feedback on outcome(s) of behavior"],
                "Social support BCT": ["3.1 Social support (unspecified)",
                                       "3.2 Social support (practical)",
                                       "3.3 Social support (emotional)"],
                "Mental representation BCT": ["4.1 Instruction on how to perform the behavior",
                                              "4.2 Information about Antecedents",
                                              "4.3 Re-attribution",
                                              "6.1 Demonstration of behavior",
                                              "13.2 Framing/reframing",
                                              "13.3 Incompatible beliefs",
                                              "13.4 Valued self-identify",
                                              "13.5 Identity associated with changed behavior"],
                "Awareness of others BCT": ["4.5. Advise to change behavior",
                                            "6.2 Social comparison",
                                            "6.3 Information about others' approval",
                                            "9.1 Credible source"],
                "Consequences BCT": ["5.1 Information about health consequences",
                                     "5.2 Salience of consequences",
                                     "5.3 Information about social and environmental consequences",
                                     "5.4 Monitoring of emotional consequences",
                                     "7.4 Remove access to the reward",
                                     "7.5 Remove aversive stimulus",
                                     "9.2 Pros and cons",
                                     "9.3 Comparative imagining of future outcomes",
                                     "10.1 Material incentive(behavior)",
                                     "10.2 Material reward(behavior)",
                                     "10.3 Non - specific reward",
                                     "10.4 Social reward",
                                     "10.5 Social incentive",
                                     "10.6 Non - specific incentive",
                                     "10.8 Incentive(outcome)",
                                     "10.9 Self - reward",
                                     "10.10 Reward(outcome)",
                                     "10.11 Future punishment",
                                     "14.1 Behavior cost",
                                     "14.2 Punishment",
                                     "14.3 Remove reward",
                                     "14.4 Reward approximation",
                                     "14.8 Reward alternative behavior",
                                     "14.9 Reduce reward frequency",
                                     "16.1 Imaginary punishment",
                                     "16.2 Imaginary reward" ],
                "External stimulus BCT": ["7.1 Prompts/cues",
                                         "7.3 Reduce prompts/cues",
                                         "7.8 Associative learning"],
                "Exposure BCT": ["7.6 Satiation",
                                 "7.7 Exposure"],
                "Habit BCT": ["8.4 Habit reversal",
                              "8.1 Behavioral practice/rehearsal",
                              "8.2 Behavior substitution",
                              "8.7 Graded tasks"],
                "Personal resources BCT": ["11.2 Reduce negative emotions",
                                           "12.6 Body changes",
                                           "13.1 Identification of self as role model",
                                           "15.1 Verbal persuasion about capability",
                                           "15.2 Mental rehearsal of successful performance",
                                           "15.3 Focus on past success",
                                           "15.4 Self - talk"
                                           ],
                "Environmental restructuring BCT": ["12.1 Restructuring the physical environment",
                                                    '12.2 Restructuring the social environment',
                                                    "12.3 Avoidance/reducing exposure to cues for the behavior",
                                                    "12.4 Distraction",
                                                    "12.5 Adding objects to the environment"]

                }


# Features that are disjoint and for which rules combining both don't make sense
# Note that in addition, each semantic domain is mutually disjoint. However,
# we don't specify these for this list of disjoints as we DO want
# rules (interactions) combining features from across semantic domains,
# while these disjoints we do not want interactions combining across them
# Note also that these features may be multiply present in the data, since
# interventions may have more than one component happening simultaneously
#
# These are specified in the direction X: [LIST_OF_Ys] where Y disjointFrom X for each Y
features_disjoint = {"placebo":["bupropion",
                                "varenicline",
                                "nrt"],
                     "Printed material": ["Digital content type",
                                          "Website / Computer Program / App",
                                          "text messaging"]
                     }

# These relationships are not hierarchical, but nevertheless they capture
# existential dependencies.
#
# These are specified in the direction X: [LIST_OF_Ys] where Y->X for each Y
features_implied = {
    "Pill": ["bupropion",
             "varenicline"],
    "11.1 Pharmacological support": [
                        "Patch",
                        "gum",
                        "e_cigarette",
                        "inhaler",
                        "lozenge",
                        "nasal_spray",
                        "placebo",
                        "nrt",
                        "Pill",
                        "bupropion",
                        "varenicline"
    ]
        }
