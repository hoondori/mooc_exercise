
from __future__ import division


# Karatuba multiplication

# find out digit number for a given positive digit
def digitNumber(x):
    if( x < 1 ):
        return 0
    else:
        return 1 + digitNumber( (x/10.0) )


# partition given number into two-part
# ex) 1234 => 12, 34
def partitionNumber(x):
    n = digitNumber(x)
    divisor = pow(10,int(n/2))
    left_part = int(x // divisor)
    right_part = int(x % divisor)
    return (left_part, right_part)

def karatuba_mulitiply(x,y):
    n = digitNumber(x)
    if( n <= 3 ):
        return x * y  # base case : simple multiplication
    else:
        a,b = partitionNumber(x)
        c,d = partitionNumber(y)
        first = karatuba_mulitiply(a,c)    # a*c
        second = karatuba_mulitiply(b,d)   # b*d
        third = karatuba_mulitiply(a+b, c+d)  # (a+b)(c+d)
        fourth = third - first - second   # ad+bc
        return int(first*pow(10,n) + fourth*pow(10,int(n/2)) + second)



print digitNumber(1234)
print partitionNumber(14801)

# x = 5678
# y = 1234
# print 'x={}, y={}, x*y={}'.format(x,y,x*y)
# print 'by karatuba_mulitiply={}'.format(karatuba_mulitiply(x,y))

x = 14801
y = 2468
print 'x={}, y={}, x*y={}'.format(x,y,x*y)
print 'by karatuba_mulitiply={}'.format(karatuba_mulitiply(x,y))

x = 56789123
y = 12341234
print 'x={}, y={}, x*y={}'.format(x,y,x*y)
print 'by karatuba_mulitiply={}'.format(karatuba_mulitiply(x,y))

x = 3141592653589793238462643383279502884197169399375105820974944592
y = 2718281828459045235360287471352662497757247093699959574966967627
print()
value1 = x*y
print 'x={}, y={}, x*y={}'.format(x,y,value1)
value2 = karatuba_mulitiply(x,y)
print 'by karatuba_mulitiply={}'.format(value2)
assert value1 == value2