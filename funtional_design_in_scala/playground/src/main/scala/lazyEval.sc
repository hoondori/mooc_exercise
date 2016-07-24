
def streamRange(lo: Int, hi:Int): Stream[Int] = {
  print(lo + " ")
  if( lo > hi ) Stream.empty
  else Stream.cons(lo, streamRange(lo+1,hi))
}

streamRange(1,10).take(3).toList

streamRange(1,10)


def from(start: Int): Stream[Int] = {
  start #:: from(start+1)
}

from(1).take(3).toList

def sieve(xs: Stream[Int]): Stream[Int] = {
  xs.head #:: sieve( xs.tail filter { x => x%xs.head != 0})
}

val primes = sieve(from(2))

primes.take(20).toList