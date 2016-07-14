
def streamRange(lo: Int, hi:Int): Stream[Int] = {
  print(lo + " ")
  if( lo > hi ) Stream.empty
  else Stream.cons(lo, streamRange(lo+1,hi))
}

streamRange(1,10).take(3).toList

streamRange(1,10)