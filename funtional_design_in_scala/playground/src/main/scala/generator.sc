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

//val booleans = integers
