

### Testing Reticulate for Python integration 

library(reticulate)
use_python("/Library/Frameworks/Python.framework/Versions/3.8/bin/python3")

os <- import("os")
os$chdir("/Users/hastingj/Work/Python/semantic-prediction/rulenn")
#os$listdir("rulenn")

#import sys; print('Python %s on %s' % (sys.version, sys.platform))
#sys.path.extend(['/Users/hastingj/Work/Python/semantic-prediction'])


#source_python('rule_nn.py')

#source_python('apply_rules.py')

source_python('ui_interface.py')


initialise_model()

# features.iloc[1].values.flatten().tolist()
features<- list(0,list(0.0,
	0.0,
	0.1,
	1.0,
	1.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	1.0,
	0.0,
	0.0,
	0.0,
	1.0,
	0.0,
	0.0,
	1.0,
	0.0,
	0.0,
	0.0,
	0.0,
	1.0,
	1.0,
	1.0,
	0.0,
	1.0,
	0.0,
	1.0,
	0.0,
	0.0,
	1.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0,
	1.0,
	1.0,
	1.0,
	1.0,
	1.0,
	0.0,
	0.0,
	0.0,
	1.0,
	0.0,
	1.0,
	0.0,
	0.0,
	0.0,
	0.0,
	0.4,
	1.0,
	0.0,
	0.0,
	0.09082684719989592,
	0.1403566070744076,
	0.37748075055338737,
	0.3880950873639953,
	0.0032407078083137976,
	0.0,
	0.0,
	0.0,
	0.0,
	0.0))

get_prediction(features)
