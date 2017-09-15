


def merge(left,right):
    left_size = len(left)
    right_size = len(right)
    total_size = left_size + right_size

    # prepare array in which sorted number will be filled
    sorted = [None] * total_size

    # iterate over two given-arrays, and smaller one will be copied to sorted array
    i = 0  # processing indicator over left array
    j = 0  # processing indicator over left array
    for k in range(total_size):
        proceed_left = None
        if( i >= left_size ): # if reached end of left array
            proceed_left = False
        elif( j >= right_size ):
            # if reached end of right array
            proceed_left = True
        elif( left[i] < right[j] ):
            # if left array element is smaller than right one
            proceed_left = True
        else:
            # if right array element is smaller than left one
            proceed_left = False

        if( proceed_left == True ):
            sorted[k] = left[i]  # copy left number to sorted array
            i = i + 1 # proceed left indicator
        else:
            sorted[k] = right[j] # copy right number to sorted array
            j = j + 1 # proceed right indicator
    return sorted

def merge_sort(arr):
    n = len(arr)
    nby2 = n/2

    if( n == 1 ): # base case
        return arr
    else:
        left_sorted = merge_sort(arr[:nby2]) # recursive call on left-half array
        right_sorted = merge_sort(arr[nby2:]) # recursive call on right-half array
        merged = merge(left_sorted,right_sorted)
        return merged


def merge_and_count_split_inv(left,right):
    left_size = len(left)
    right_size = len(right)
    total_size = left_size + right_size

    # prepare array in which sorted number will be filled
    sorted = [None] * total_size

    # initialize inversion count
    inv = 0

    # iterate over two given-arrays, and smaller one will be copied to sorted array
    i = 0  # processing indicator over left array
    j = 0  # processing indicator over left array
    for k in range(total_size):
        proceed_left = None
        if( i >= left_size ): # if reached end of left array
            proceed_left = False
        elif( j >= right_size ):
            # if reached end of right array
            proceed_left = True
        elif( left[i] < right[j] ):
            # if left array element is smaller than right one
            proceed_left = True
        else:
            # if right array element is smaller than left one
            proceed_left = False

        if( proceed_left == True ):
            sorted[k] = left[i]  # copy left number to sorted array
            i = i + 1 # proceed left indicator
        else:
            sorted[k] = right[j] # copy right number to sorted array
            j = j + 1 # proceed right indicator

            # only if right element is copied to soreted array, we found some inversion
            inv = inv + (left_size - i)

    return sorted,inv

def merge_sort_and_count(arr):
    n = len(arr)
    nby2 = n/2

    if( n == 1 ): # base case
        return arr,0
    else:
        left_sorted,left_cnt = merge_sort_and_count(arr[:nby2]) # recursive call on left-half array
        right_sorted,right_cnt = merge_sort_and_count(arr[nby2:]) # recursive call on right-half array
        sorted,split_cnt = merge_and_count_split_inv(left_sorted,right_sorted)
        return sorted,left_cnt+right_cnt+split_cnt


print merge_sort([5,4,1,8,7,2,6,3])

print merge_sort([1,3,5,2,4,6])

print merge_sort_and_count([1,3,5,2,4,6])   # inversion = 3

with open('IntegerArray.txt') as f:
    arrStr = f.read().splitlines()
    arrInt = [int(x) for x in arrStr]

    sorted, inv = merge_sort_and_count(arrInt)
    print inv

