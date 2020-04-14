import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../bin')))

import pymlip

print('Setting pass test')


mlp = pymlip.pymlip('test')

st={'mlip':'mtp','calculate-efs':'true','log':'stdout'}

st1 = pymlip.settings()

st1['mlip']='mtp'
st1['mlip:load-from']='fitted.mtp'
st1['write-efs']='true'
st1['log']='stdout'


mlp.pass_settings(st1())

status=os.EX_OK

sys.exit(status)
