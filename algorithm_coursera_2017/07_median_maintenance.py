
import heapq


### https://stackoverflow.com/questions/2501457/what-do-i-use-for-a-max-heap-implementation-in-python

class MaxHeapObj(object):
  def __init__(self,val): self.val = val
  def __lt__(self,other): return self.val > other.val
  def __eq__(self,other): return self.val == other.val
  def __str__(self): return str(self.val)

class MinHeap(object):
  def __init__(self): self.h = []
  def heappush(self,x): heapq.heappush(self.h,x)
  def heappop(self): return heapq.heappop(self.h)
  def __getitem__(self,i): return self.h[i]
  def __len__(self): return len(self.h)

class MaxHeap(MinHeap):
  def heappush(self,x): heapq.heappush(self.h,MaxHeapObj(x))
  def heappop(self): return heapq.heappop(self.h).val
  def __getitem__(self,i): return self.h[i].val

minh = MinHeap()
maxh = MaxHeap()
# add some values
# minh.heappush(12)
# maxh.heappush(12)
# minh.heappush(4)
# maxh.heappush(4)
# print len(minh)
# print len(maxh)
# # fetch "top" values
# print(minh[0],maxh[0]) # "4 12"
# # fetch and remove "top" values
# print(minh.heappop(),maxh.heappop()) # "4 12"



f = open('./Median.txt','r')
#f = open('./Median_small.txt','r')
lines = f.readlines()
numbers = []
for line in lines:
    numbers.append(int(line))

print numbers

median = []
for i in numbers:
    print "x", i
    # insert x into either of minh or maxh
    if( len(maxh) == 0 or i < maxh[0]):
        print "push to maxh"
        maxh.heappush(i) # add to max heap
    else:
        print "push to minh"
        minh.heappush(i) # otherwise, add to min heap


    # balance two heap
    if( len(maxh) - len(minh) >= 2 ):
        minh.heappush(maxh.heappop())
    elif( len(minh) - len(maxh) >= 2 ):
        maxh.heappush(minh.heappop())

    print "minh.size", len(minh)
    print "maxh.size", len(maxh)

    # decide median
    if( len(minh) > len(maxh)):
        print "meadian is minh[0", minh[0]
        median.append(minh[0])
    else:
        print "meadian is maxh[0]", maxh[0]
        median.append(maxh[0])

print "median", median
print "median len",len(median)
print "sum of median", sum(median)
