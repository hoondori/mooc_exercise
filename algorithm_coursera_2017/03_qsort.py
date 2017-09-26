
import random

# for choose pivot randomly
random.seed()

# keep track of how many comparision of two elements be done
g_comparision_count = 0


def qsort(arr, low, high):
    # print("called: ", arr[low:high])

    # base case
    if (low >= high or len(arr) == 0):
        return

    # Choose good pivot
    #p = choose_pivot_first(low, high)
    #p = choose_pivot_last(low, high)
    #p = choose_pivot_random(low, high)
    p = choose_pivot_median_of_median(arr,low,high)
    # print("pivot:", p, arr[p])

    # partition array around pivot
    left, right = partition_around_pivot(arr, low, high, p)

    # recursive call on left and right
    # print("s",left,right)
    qsort(arr, low, left)
    qsort(arr, right, high)


def choose_pivot_first(low, high):
    # naive choose first one
    return low

def choose_pivot_last(low, high):
    # naive choose last one
    return high - 1

def choose_pivot_random(low, high):
    # random choise lead to good pivot on average
    if( low == high - 1 ):
        return low
    else:
        print(low,high)
        return random.randrange(low,high-1)


# you should choose the pivot as follows.
# Consider the first, middle, and final elements of the given array.
# (If the array has odd length it should be clear what the "middle" element is;
# for an array with even length 2k, use the kth element as the "middle" element.
# So for the array 4 5 6 7, the "middle" element is the second one ... 5 and not 6!)
# Identify which of these three elements is the median
def choose_pivot_median_of_median(arr,low,high):
    # number of element
    n = high - low

    # determine location of middle
    m = (n/2)-1 if (n % 2 == 0) else n/2

    first = arr[low]
    last = arr[high-1]
    middle = arr[m]

    ordered = sorted([(first,low),(last,high-1),(middle,m)])
    return ordered[1][1]

def partition_around_pivot(arr, low, high, p):

    # update comparision count
    global g_comparision_count
    g_comparision_count = g_comparision_count + (high - low - 1)

    # make sure pivot is in first position
    if (p != low):
        # swap
        temp = arr[p]
        arr[p] = arr[low]
        arr[low] = temp

    # sweep through, with i,j index
    # if unpartitioned element is less than pivot,
    # then swap it with first element of greater, and then proceed i index one-step ahead
    i = low + 1
    for j in range(low + 1, high):
        if (arr[j] < arr[low]):
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
            i = i + 1

    # finally locate pivot to the right position
    # swap first element with last element of less-than
    temp = arr[i - 1]
    arr[i - 1] = arr[low]
    arr[low] = temp
    return i - 1, i


with open('QuickSort.txt') as f:
    arrStr = f.read().splitlines()
    arrInt = [int(x) for x in arrStr]

    g_comparision_count = 0
    print(arrInt[0:20])
    qsort(arrInt, 0, len(arrInt))
    print(arrInt[0:100])
    print(g_comparision_count)