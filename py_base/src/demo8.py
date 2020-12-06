#coding=utf-8
class Solution():
    def full_package(self,goods,max_V):
        print("pre:",goods)
        self.goods_remove(goods)
        print("aft",goods)
        goods_add = []
        for i in goods:
            k = 1
            while i[0]*(2**k)<=max_V:
                goods_add.append([i[0]*(2**k),i[1]*(2**k)])
                k+=1
        goods.extend(goods_add)
        print("goods_add 后:",goods)
        f = [0] * (max_V + 1)
        for i in range(len(goods)):
            for v in range(max_V,goods[i][0]-1,-1):
                #特别注意这个ｉｆ语句，如果不写，结果可能会出现错误！！！
                if v>=goods[i][0]:
                    f[v] = max(f[v],f[v-goods[i][0]]+goods[i][1])
        print(f)
        return(f[max_V])

    def goods_remove(self,goods):
        # 操作简化，将所有的不符合条件的good删除
        good_delete = set([])
        for i in range(len(goods)):
            if i in good_delete: continue
            for j in range(len(goods)):
                if goods[i][0] <= goods[j][0] and goods[i][1] >= goods[j][1]:
                    if j == i: continue
                    good_delete.add(j)
                elif goods[i][0] >= goods[j][0] and goods[i][1] <= goods[j][1]:
                    if j == i: continue
                    good_delete.add(i)
                else:continue
        good_delete = list(good_delete)
        good_delete.sort(reverse=True)
        for i in good_delete:
            del goods[i]



goods = [[5,12],[4,3],[7,10],[2,3],[6,6]]
test = Solution()
print(test.full_package(goods,16))