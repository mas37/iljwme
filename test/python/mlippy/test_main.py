import os

total=3
failed=0

ncores = 1

captions = ['Training test',
			'Selection test',
			'Relaxation test']

for i in range (1,total+1):
	print('test ' + str(i) +' started ('+captions[i-1]+')')

	flag=999
	
	os.system('mkdir out')
	#os.system('mpiexec -n ' + str(ncores) + ' ./' + str(i) +'/test_bin.sh >> /dev/null')
	#flag = os.system('mpiexec -n ' + str(ncores) + ' python ' + str(i) +'/test.py >> /dev/null')
	os.system('chmod +x ' + './' + str(i) + '/*.sh')
	os.system(' ./' + str(i) +'/test_bin.sh >> /dev/null')
	flag = os.system('python ' + str(i) +'/test.py >> /dev/null')

	if (flag==os.EX_OK):
		print ('...sucess')
	else:
		print ('...failed')
		failed+=1

	os.system('rm -rf out')

print(str(failed)+' tests failed out of ' + str(total))

