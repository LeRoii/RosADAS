import sys

n = int(sys.stdin.readline().strip())
print(n)
for i in range(n):
    line = sys.stdin.readline().strip()
    print(line.split(','))
    values = list(map(int,line.split(',')))
# print(values)