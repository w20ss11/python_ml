#coding=utf-8
class Solution():
    def full_package(self,goods,max_V):
        print("delete前   ",goods)
        self.goods_remove(goods)
        print("delete后   ",goods)
        f = [0] * (max_V + 1)
        for i in range(len(goods)):
            for v in range(max_V+1):
                if v>=goods[i][0]:
                    f[v] = max(f[v],f[v-goods[i][0]]+goods[i][1])
        print(f)
        return f[max_V]

    def goods_remove(self,goods):
        good_delete = set([])
        for i in range(len(goods)):
            if i in good_delete:
                continue
            for j in range(len(goods)):
                if goods[i][0] <= goods[j][0] and goods[i][1] >= goods[j][1]:
                    if j == i: continue
                    good_delete.add(j)
                elif goods[i][0] >= goods[j][0] and goods[i][1] <= goods[j][1]:
                    if j == i: continue
                    good_delete.add(i)
                else:
                    continue
        good_delete = list(good_delete)
        good_delete.sort(reverse=True)
        for i in good_delete:
            del goods[i]



goods = [[5,12],[4,3],[7,10],[2,3],[6,6]]
test = Solution()
print(test.full_package(goods,16))