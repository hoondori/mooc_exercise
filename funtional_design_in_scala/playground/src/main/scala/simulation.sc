
trait Simulation {
  type Action = () => Unit

  case class Event(time: Int, action: Action)
  class Wire {
    private var sigVal = false
    private var actions: List[Action] = List()
    def getSignal: Boolean = sigVal
    def setSignal(s: Boolean): Unit = {
      if(s != sigVal) {
        sigVal = s
        actions foreach (_())
      }
    }
    def addAction(a: Action): Unit = {
      actions = a :: actions
      a()
    }
  }

  private type Agenda = List[Event]
  private var agenda: Agenda = List()
  private var curTime = 0

  def currentTime: Int = curTime

  def afterDelay(delay: Int)(block: => Unit): Unit = {
    val item = Event(currentTime+delay, () => block)
    agenda = insert(agenda, item)
  }

  private def insert(ag: List[Event], item:Event): List[Event] = {
    ag match {
      case first :: rest if (first.time <= item.time) => {
        first :: insert(rest, item)
      }
      case _ => item :: ag
    }
  }

  private def loop(): Unit = {
    agenda match {
      case first::rest => {
        agenda = rest
        curTime = first.time
        first.action()
        loop()
      }
      case Nil =>
    }
  }

  def run(): Unit = {
    afterDelay(0) {
      println("*** simulation started, time = " + currentTime + " ***")
    }
    loop()
  }

  def probe(name: String, wire: Wire): Unit = {
    def probeAction(): Unit = {
      println(s"$name $currentTime value = ${wire.getSignal}")
    }
    wire addAction probeAction
  }


}

abstract class Gates extends Simulation {

  def InverterDelay:Int
  def AndGateDelay:Int
  def OrGateDelay:Int

  def inverter(input: Wire, output: Wire): Unit = {
    def invertAction(): Unit = {
      val inputSig = input.getSignal
      afterDelay(InverterDelay) {
        output setSignal !inputSig
      }
    }
    input addAction invertAction
  }
  def andGate(a1: Wire, a2: Wire, output: Wire): Unit = {
    def andAction(): Unit = {
      val inSig1 = a1.getSignal
      val inSig2 = a2.getSignal
      afterDelay(AndGateDelay) {
        output setSignal (inSig1 & inSig2)
      }
    }
    a1 addAction andAction
    a2 addAction andAction
  }
  def orGate(o1: Wire, o2: Wire, output: Wire): Unit = {
    def orAction(): Unit = {
      val inSig1 = o1.getSignal
      val inSig2 = o2.getSignal
      afterDelay(OrGateDelay) {
        output setSignal (inSig1 | inSig2)
      }
    }
    o1 addAction orAction
    o2 addAction orAction
  }

}

abstract class Circuits extends Gates {

  def halfAdder(a: Wire, b: Wire, s:Wire, c: Wire): Unit = {
    val d = new Wire
    val e = new Wire
    orGate(a,b,d)
    andGate(a,b,c)
    inverter(c,e)
    andGate(d,e,s)
  }

  def fullAdder(a: Wire, b: Wire, cin: Wire, sum: Wire, cout: Wire): Unit = {
    val s,c1,c2 = new Wire
    halfAdder(b,cin,s,c1)
    halfAdder(a,s,sum,c2)
    orGate(c1,c2,cout)
  }
}

trait Parameters {
  def InverterDelay = 2
  def AndGateDelay = 3
  def OrGateDelay = 5
}

object sim extends Circuits with Parameters

import sim._

val in1, in2, sum, carry = new Wire

halfAdder(in1,in2,sum,carry)

probe("sum", sum)
probe("carry", carry)

in1 setSignal true
run()

in2 setSignal true
run()

