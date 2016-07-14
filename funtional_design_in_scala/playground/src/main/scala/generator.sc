println("bab")

trait Generator[+T] {
  self =>

  def generate: T

  def map[S](f: T =>S): Generator[S] = new Generator[S] {
    def generate = f(self.generate)
  }

  def flatMap[S](f: T => Generator[S]):Generator[S] = new Generator[S] {
    def generate: S = f(self.generate).generate
  }
}
val integers = new Generator[Int] {
  val rand = new java.util.Random()
  def generate = rand.nextInt()
}

println(integers.generate)

val booleans = integers map { x=> x > 0 }

def pairs[T,U](t:Generator[T], u:Generator[U]) = {
  t flatMap { x =>
    u map { y =>
      (x,y)
    }
  }
}

def single[T](x:T): Generator[T] = new Generator[T] {
  override def generate: T = x
}

def choose(lo: Int, hi:Int): Generator[Int] = {
  for( x <- integers ) yield lo + x % (hi-lo)
}

def oneOf[T](xs: T*): Generator[T] = {
  for( idx <- choose(0, xs.length) ) yield xs(idx)
}

def lists: Generator[List[Int]] = for {
  isEmpty <- booleans
  list <- if(isEmpty) emptyLists else nonEmptyLists
} yield list

def emptyLists = single(Nil)

def nonEmptyLists = for {
  head <- integers
  tail <- lists
} yield head::tail

//////////////////////////////////////////////////
// Tree Generator

trait Tree
case class Inner(left:Tree, right:Tree) extends Tree
case class Leaf(x: Int) extends Tree

def leafs: Generator[Leaf] = for {
  x <- integers
} yield Leaf(x)

def inners: Generator[Inner] = for {
  left <- trees
  right <- trees
} yield Inner(left,right)

def trees: Generator[Tree] = for {
  isLeaf <- booleans
  tree <- if(isLeaf) leafs else inners
} yield tree

println(trees.generate)

//////////////////////////////////////////////////
// Random input testing

def test[T](r: Generator[T], noTimes: Int)(f: T => Boolean) = {

  for( i <- 0 until noTimes) {
    val pick = r.generate
    val result = f(pick)
    assert( f(pick), "test failed for value: " + pick )
  }
  println(s" passed ${noTimes}")
}

test(pairs(lists,lists),100) {
  case (xs,ys) => (xs++ys).length > xs.length
}


