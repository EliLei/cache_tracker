import torch
import collections

# https://blog.csdn.net/Sableye/article/details/114304084

class ARC:
    def __init__(self, max_size):
        self.c = max_size
        self.p = 0.0
        self.T1 = collections.OrderedDict()
        self.T2 = collections.OrderedDict()
        self.B1 = collections.OrderedDict()
        self.B2 = collections.OrderedDict()

    def update(self, key, new_key,value):
        if key!=None and key!=new_key:
            new_key=key+'/'+new_key

        if key in  self.T1:
            hit=True
            self.T1.pop(key)
            self.T2[new_key] = value

        elif key in self.T2:
            hit=True
            self.T2.pop(key)
            self.T2[new_key] = value

        elif key in self.B1:
            hit=False
            delta = max(1., len(self.B2)/len(self.B1))
            self.p = min(self.p+delta, self.c)
            self.replace(key, new_key, value)
            self.B1.pop(key)
            self.T2[new_key]=value

        elif key in self.B2:
            hit=False
            delta = max(1., len(self.B1)/len(self.B2))
            self.p = max(self.p-delta, 0)
            self.replace(key, new_key, value)
            self.B2.pop(key)
            self.T2[new_key]=value

        elif len(self.T1) + len(self.B1) == self.c:
            hit = False
            if len(self.T1) < self.c:
                self.B1.popitem(last=False)
                self.replace(key, new_key, value)
            else:
                self.T1.popitem(last=False)
            self.T1[new_key]=value

        elif len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) >= self.c:
            hit = False
            if len(self.T1) + len(self.B1) > self.c:
                # error L1 exceeds c
                pass
            if len(self.T1)+len(self.T2) != self.c:
                # T1 + T2 >=c
                pass
            else:
                pass
            if len(self.T1)+len(self.T2)+len(self.B1)+len(self.B2)==self.c*2:
                self.B2.popitem(last=False)
            self.replace(key, new_key, value)
            self.T1[new_key]=value
        else:
            hit=False
            self.T1[new_key] = value


    def replace(self, key, new_key, value):
        if len(self.T1) > 0 and (len(self.T1)> self.p or (key in self.B2)):
            del_k, del_v = self.T1.popitem(last=False) # FIFO
            self.B1[del_k] = del_v
        else:
            del_k, del_v = self.T2.popitem(last=False)  # FIFO
            self.B2[del_k] = del_v

    def getall(self):
        return {**self.T1, **self.T2, **self.B1, **self.B2}

    def get(self):
        return {**self.T1, **self.T2}

if __name__ == '__main__':
    def printarc(arc):
        print('T1', arc.T1)
        print('T2', arc.T2)
        print('B1', arc.B1)
        print('B2', arc.B2)
        print()

    def arctest(size, randint, iter):
        import random
        arc = ARC(size)
        for i in range(iter):
            key=random.randint(1,randint)
            v=1

            print('visit',key)
            arc.update(key,key,v)
            printarc(arc)
            pass



    arctest(4, 18, 100)

