
def karatsuba(x, y):
    """Function to multiply 2 numbers in a more efficient manner than the grade school algorithm"""
    if len(str(x)) == 1 or len(str(y)) == 1:
        return x * y
    else:
        n = max(len(str(x)), len(str(y)))
        nby2 = n / 2

        a = x / 10 ** (nby2)
        b = x % 10 ** (nby2)
        c = y / 10 ** (nby2)
        d = y % 10 ** (nby2)

        ac = karatsuba(a, c)
        bd = karatsuba(b, d)
        ad_plus_bc = karatsuba(a + b, c + d) - ac - bd

        # this little trick, writing n as 2*nby2 takes care of both even and odd n
        prod = ac * 10 ** (2 * nby2) + (ad_plus_bc * 10 ** nby2) + bd

        return prod

def test(x,y):
    val1 = x*y
    val2 = karatsuba(x,y)
    print 'x={}, y={}, x*y={}'.format(x, y, val1)
    print 'by karatuba_mulitiply={}'.format(karatsuba(x, y))
    assert val1 == val2


test(14801,2468)
test(3141592653589793238462643383279502884197169399375105820974944592, 2718281828459045235360287471352662497757247093699959574966967627)
