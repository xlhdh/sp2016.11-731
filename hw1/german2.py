a = [s for s in open("germanwords.txt")]
b = [s for s in open("germancons.txt")]
for i in range(len(a)):
	if not any(c.isdigit() for c in a[i]): 
		print a[i].strip()+"="+"+".join(b[i].strip().split(','))