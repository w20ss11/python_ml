# coding=gbk 
print('hello')
print(2+2)

#x=input('x:')
#print('x is',x)

if(1==2):print('haha')
else:print('中文heihei')

f=open('haha.txt','a')
f.write('hahaa')
f.close()

f=open('haha.txt','r')
print('=====part1=======')
print(f.read())
f.close()

print('=====part2=======')
f=open('haha.txt')
while True:
    line=f.readline()
    if not line:break
    print(line)
f.close()

print('=====part3=======')
f=open('haha.txt')
for line in f.readlines():
    print(line)
f.close()

print('=====part4=======')
for line in open('haha.txt'):
    print(line)



