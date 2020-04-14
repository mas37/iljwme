import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../bin')))

import pymlip

print('mtpr training test')


mlp = pymlip.pymlip('test1')

mlp.load_potential("pot.mtp")
mlp.load_training_set("train.cfg")
mlp.load_validation_set("valid.cfg")

if not os.path.exists('./out'):
	os.makedirs('./out')
else: 
	if os.path.isfile("./out/trained.mtp"):
		os.remove('./out/trained.mtp')

mlp.train(1e-2,1e-3,4,1e-8)
#mlp.get_train_errors()
#mlp.get_validation_errors()
mlp.save_potential("out/trained.mtp")

status=os.EX_OK

if (os.path.isfile("./out/trained.mtp")==False):
	status=1

sys.exit(status)
	
