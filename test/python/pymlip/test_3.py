import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../bin')))

from pymlip import *

print('mtpr selection test')

mlp = pymlip('test3')

if not os.path.exists('./out'):
	os.makedirs('./out')
else: 
	if os.path.isfile("./out/diff.cfg"):
		os.remove('./out/diff.cfg')

mlp.select_add('Trained.mtp','train.cfg','valid.cfg','./out/diff.cfg')

status=os.EX_OK

if (os.path.isfile('./out/diff.cfg')==False):
	status=1

st = settings()

st['mlip']='mtp'
st['mlip:load-from']='fitted.mtp'
st['write-efs']='true'
st['log']='stdout'

mlp.pass_settings(st())

sys.exit(status)
