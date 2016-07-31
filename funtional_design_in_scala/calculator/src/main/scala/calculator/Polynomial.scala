package calculator

object Polynomial {
  def computeDelta(a: Signal[Double], b: Signal[Double],
      c: Signal[Double]): Signal[Double] = {
    Signal {
      val aVal = a()
      val bVal = b()
      val cVal = c()

      bVal*bVal - 4*aVal*cVal
    }

  }

  def computeSolutions(a: Signal[Double], b: Signal[Double],
      c: Signal[Double], delta: Signal[Double]): Signal[Set[Double]] = {

    Signal {
      val aVal = a()
      val bVal = b()
      val cVal = c()
      val deltaVal = delta()

      if( deltaVal >= 0 && a() != 0) {
        Set(
          (-1*bVal + scala.math.sqrt(deltaVal))/(2*aVal),
          (-1*bVal - scala.math.sqrt(deltaVal))/(2*aVal)
        )
      } else {
        Set()
      }
    }
  }
}
