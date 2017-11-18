import urllib

ifile = open("../../Data/item_urls.txt","r")
ofile = open("../../Data/outliers.txt","w")
i = 0
for l in ifile:
	i+=1
	if i%1000==0:
		print i
	ll = l
	if len(l.split(","))>2:
		ofile.write(ll)
		continue
	else:
		name, url = l.split(",")
	name = "../../Data/images/"+name+".jpg"
	url = url[0:-1]
	urllib.urlretrieve(url, name)
ifile.close()