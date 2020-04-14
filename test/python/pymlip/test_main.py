import os

total=3
failed=0

captions = ['mtpr training test',
			'configurations reading/writing test',
			'mtpr selection test']

for i in range (1,total+1):
	print('test ' + str(i) +' started ('+captions[i-1]+')')

	flag=999

	if (i==1):
		flag = os.system('mpiexec -n 4 python test_'+str(i)+'.py >> /dev/null')
	else:
		flag = os.system('python test_'+str(i)+'.py >> /dev/null')

	if (flag==os.EX_OK):
		print ('...sucess')
	else:
		print ('...failed')
		failed+=1

print(str(failed)+' tests failed out of ' + str(total))

