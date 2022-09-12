#  RoundingCleaner
*Description:*  Casts values to numbers and rounds them to at most 3 decimal places. If
there are multiple possible values, we attempt to find a rounding function
that unifies all - e.g. ["2.51", "2.49999999"] will be unified and rounded
to 2.5.

This cleaner is applied to the following features: 6080481, 6080485, 6080486, 6080512, 6080719

#  HealthProfessionalCleaner
*Description:*  Classifies as any of the following classes as "present" if any of assiciated values noted below is matched by a value:
  * nurse:  "nurse"
  * doctor: "physician", "doctor", "physician", "cardiologist", "pediatrician", "general pract", "GP", "resident", "internal medicine"

This cleaner is applied to the following features: 6080704

#  DigitalContentCleaner
*Description:*  Classifies as "present" if any of the values contains anything that
matches "text" or "text message".

This cleaner is applied to the following features: 6080691

#  CountryCleaner
*Description:*  Countries are currently omitted

This cleaner is applied to the following features: 6080518

#  PregnancyTrialCleaner
*Description:*  The kind of trial is encoded as follows:
1 -> Pregnancy trial
2 -> Pregnancy trial (Mixed)

Individual features are introduced for each of these classes.

This cleaner is applied to the following features: 

#  MotivationalIntervewingCleaner
*Description:*  Classifies as "present" if any of the values contains anything that
matches "brief advise" or "ba".

This cleaner is applied to the following features: 6823487

#  OutcomeValueCleaner
*Description:*   Outcome values that merges the values of the outcomes in the JSON-file
and some manual corrections. If there are manual corrections, those take
precedence over the JSON data.
    

This cleaner is applied to the following features: 6451791

#  PresenceCleaner
*Description:*   Any value will be considered as "presence".

This cleaner is applied to the following features: 6451788, 6823485, 6823485, 6452745, 6452746, 6452747, 6452748, 6452749, 6452750, 6452751, 6452752, 6452753, 6452755, 6452756, 6452757, 6452758, 6452759, 6452760, 6452761, 6452763, 6452764, 6452765, 6452831, 6452832, 6452833, 6452836, 6452838, 6452839, 6452840, 6452843, 6452847, 6452848, 6452849, 6452851, 6452853, 6452854, 6452855, 6452856, 6452857, 6452858, 6452860, 6452861, 6452863, 6452930, 6452932, 6452933, 6452934, 6452936, 6452937, 6452938, 6452939, 6452940, 6452941, 6452943, 6452944, 6452945, 6452946, 6452948, 6452949, 6452954, 6452955, 6452956, 6452957, 6452959, 6452960, 6452962, 6452963, 6452964, 6452965, 6452966, 6452968, 6452969, 6452970, 6452973, 6452977, 6452978, 6452981, 6452982, 6452983, 6452984, 6452986, 6452987, 6080701, 6080686, 6080692, 6080694, 6080688, 6830264

#  SomaticCleaner
*Description:*  Classifies as any of the following classes as "present" if any of assiciated values noted below is matched by a value:
  * gum:         "gum", "polacrilex"
  * lozenge:     "lozenge"
  * e_cigarette: "ecig", "ecigarette"
  * inhaler:     "inhaler", "inhal"
  * placebo:     "placebo"
  * varenicline: "varenicline", "varen", "chantix", "champix"
  * nasal_spray: "nasal"
  * rimonabant:  "rimonab"
  * nrt:         "nicotine", "nrt"
  * bupropion:   "bupropion"

This cleaner is applied to the following features: 6080693

#  HealthCareFacilityCleaner
*Description:*  <bound method HealthCareFacilityCleaner.docs of <__main__.HealthCareFacilityCleaner object at 0x7ff85236a0e0>>

This cleaner is applied to the following features: 6080629

#  TimePointCleaner
*Description:*  Values from "Combined follow up" are extracted and, if present, overwritten
by manual changes. Manual annotations are split into unit and value.
The values are then normalised to "weeks" using the following factors:
* days:   1/7
* weeks:  1
* months: 4.35
* year:   12 * 4.35

This cleaner is applied to the following features: 6451782

#  DistanceCleaner
*Description:*  Classifies as "present" if any of the values contains anything that
matches "phone", "call", "telephone", "quitline" or "hotline".

This cleaner is applied to the following features: 6080687

#  PatientRoleCleaner
*Description:*  Classifies as any of the following classes as "present" if any of assiciated values noted below is matched by a value:
  * aggregate patient role: "patient"

This cleaner is applied to the following features: 6080508

#  PillCleaner
*Description:*  Looks for the following keywords:
* bupropion:     "bupropion"
* nortriptyline: "nortriptyline"
* varenicline:   "varenicline", "varen", "chantix", "champix"

For bupropion, it is also checked whether 6080693 contains "bupropion".
    

This cleaner is applied to the following features: 6080695

#  PsychologistCleaner
*Description:*  Classifies as any of the following classes as "present" if any of assiciated values noted below is matched by a value:
  * pychologist: "pychologist", "psychol"

This cleaner is applied to the following features: 6080706

#  PharmacologicalInterestCleaner
*Description:*  Any value in 6830268 or 6830264 will be considered as "presence".

This cleaner is applied to the following features: 6830268

#  RelapsePreventionTrialCleaner
*Description:*  The kind of trial is encoded as follows:
1 -> Relapse Prevention Trial
2 -> Relapse Prevention Trial (Mixed)

Individual features are introduced for each of these classes.

This cleaner is applied to the following features: 