/**
  * Created by Anthony on 12/19/15.
  *
  */


import java.io.{File, PrintWriter}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.immutable.HashSet


package object frequent {
  val rootDirectory = "./"

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[3]")
      .setAppName("FrequentItemsetMining")
    val sc = new SparkContext(conf)
    LogManager.getRootLogger.setLevel(Level.WARN)

    println("\nReading train...")
    val mappedRdd = tidyInputData(sc, loadData(sc, "train.csv"))
    mappedRdd.cache()

    println("\nFrequent item sets:")
    val itemsets = computeFrequentItemSets(mappedRdd).filter(_.items.length > 2).collect
    for (i <- itemsets.take(50)) {println(i.items.toSeq.sortWith(_ < _).mkString(", "))}
    println("\tFound " + itemsets.count(_ => true).toString)
    val tripTypes = itemsets.map(_.items.toSeq.sortWith(_ < _).head).distinct

    tripTypes foreach println

    val itemSets = itemsetsSortedByTripType(itemsets)
    writeItemSets(itemSets)
  }


  // Load the input source data to compute frequent itemsets
  def loadData(sc: SparkContext, filename: String): DataFrame = {
    new SQLContext(sc).read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(rootDirectory + "input/" + filename)
  }






  // Map the RDD to contain only an array of items per visit
  def tidyInputData(sc: SparkContext, df: DataFrame): RDD[(Int, Set[String])] = {
    def reshapeRow(r: Row): (Int, Set[String]) = {
      val (tripTypeIx, visitNumberIx, departmentDescriptionIx, finelineNumberIx) = (0, 1, 5, 6)

      val visitNumber = r.getInt(visitNumberIx)
      val tripType = r.getInt(tripTypeIx)
      val finelineNumber = r.getInt(finelineNumberIx)
      val departmentDescription = r.getString(departmentDescriptionIx)
      (visitNumber, HashSet((-1 * tripType).toString) ++ HashSet(departmentDescription))
    }

    // These trip types can be ignored
    val ignoreTypes: Set[String] = Set("-8", "-39", "-9", "-999", "-40")

    df.rdd
      .map((r: Row) => reshapeRow(r))
      .reduceByKey(_ ++ _)
      .filter((f: (Int, Set[String])) => (ignoreTypes & foo._2).isEmpty)
  }

  def computeFrequentItemSets(rdd: RDD[(Int, Set[String])]): RDD[FreqItemset[String]] = {
    val fpg = new FPGrowth()
    fpg.setMinSupport(0.0004)
    val model = fpg.run(rdd.map(_._2.toArray).cache())
    model.freqItemsets.filter(_.items.exists(_.head == '-'))
  }


  case class ItemSet(tripType: Int, items: Array[String])

  def itemsetsSortedByTripType(itemsets: Array[FreqItemset[String]]): Array[ItemSet] = {
    val sortedItemSets: Array[Array[String]] = itemsets.map(_.items.sortWith(_ < _))
    def cleanupTripType(s: Array[String]): ItemSet = {
      val items = s.tail
      val tripType = s.head.toInt * -1
      ItemSet(tripType, items)
    }

    sortedItemSets.map(cleanupTripType)
  }




  def writeItemSets(itemSets: Array[ItemSet]): Unit = {
    println("\nWriting itemsets...")
    val pw = new PrintWriter(new File(rootDirectory + "cleaned/" + "itemsets.txt"))

    for {
      itemSet <- itemSets
      row = itemSet.tripType.toString +: itemSet.items
    } {pw.println(row.mkString(";"))}

      pw.close()

  }

}
