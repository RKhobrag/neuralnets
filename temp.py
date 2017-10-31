def f():
	temp = open('temp.csv', 'w')
	for i in range(0,200000):
		temp.write(str(i))
		temp.write("\n")
f()
