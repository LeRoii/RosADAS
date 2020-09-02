def maxProfit(prices):
    n = len(prices)
    if n < 2:
        return 0
    dp1 = [0 for _ in range(n)]
    dp2 = [0 for _ in range(n)]
    minval = prices[0]
    maxval = prices[-1]
    # 前向
    for i in range(1, n):
        dp1[i] = max(dp1[i - 1], prices[i] - minval)
        minval = min(minval, prices[i])
    # 后向
    for i in range(n - 2, -1, -1):
        dp2[i] = max(dp2[i + 1], maxval - prices[i])
        maxval = max(maxval, prices[i])

    dp = [dp1[i] + dp2[i] for i in range(n)]
    print(max(dp))
    return max(dp)
#
# N = input()
# c=[]
# # print(N[1:-1].split(','))
# for i in N[1:-1].split(','):
#     c.append(int(i))
# # print(c)
# maxProfit(c)
import sys
input = []
try:
    # while True:
    line = sys.stdin.readline().strip()
    print(line)
        # if line == "":
        #     break
        # lines = list(map(int, line.split(',')))
        # input.append(lines)
except:
    pass
for i in line[1:-1].split(','):
    input.append(int(i))
print(input)
maxProfit(input)

# print(input)
