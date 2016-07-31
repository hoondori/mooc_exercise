package calculator

object TweetLength {
  final val MaxTweetLength = 140

  def tweetRemainingCharsCount(tweetText: Signal[String]): Signal[Int] = {
    Signal {
      MaxTweetLength - tweetLength(tweetText())
    }
  }

  def colorForRemainingCharsCount(remainingCharsCount: Signal[Int]): Signal[String] = {

    /*If there are 15 or more characters left, the color "green"
    If there are between 0 and 14 characters left, included, the color "orange"
    Otherwise (if the remaining count is negative), the color "red"*/

    Signal {
      remainingCharsCount() match {
        case cnt if cnt >= 15 => "green"
        case cnt if cnt >=0 && cnt <= 14 => "orange"
        case _ => "red"
      }
    }
  }

  /** Computes the length of a tweet, given its text string.
   *  This is not equivalent to text.length, as tweet lengths count the number
   *  of Unicode *code points* in the string.
   *  Note that this is still a simplified view of the reality. Full details
   *  can be found at
   *  https://dev.twitter.com/overview/api/counting-characters
   */
  private def tweetLength(text: String): Int = {
    /* This should be simply text.codePointCount(0, text.length), but it
     * is not implemented in Scala.js 0.6.2.
     */
    if (text.isEmpty) 0
    else {
      text.length - text.init.zip(text.tail).count(
          (Character.isSurrogatePair _).tupled)
    }
  }
}
