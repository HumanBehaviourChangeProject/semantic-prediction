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
                        "11.2 Reduce negative emotions"
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
                        "placebo",
                        "nrt",
                        "Pill",
                        "bupropion",
                        "varenicline",
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
                        "e_cigarette",
                        "inhaler",
                        "lozenge",
                        "nasal_spray",
                        "placebo",
                        "nrt",
                        "Pill",
                        "bupropion",
                        "varenicline"],
                "nrt": ["e_cigarette",
                        "bupropion",
                        "varenicline"],
                "Abstinence type": [
                        "Abstinence: Continuous",
                        "Abstinence: Point Prevalence"],
                "Health Professional": [
                        "doctor",
                        "nurse"	,
                        "pychologist"
                    ],
                "Digital content type": ["Website / Computer Program / App",
                                         "text messaging"]
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