package com.bigdata.spark

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}

object KmansFromHive {

  LoggerLevels.setStreamingLogLevels()

  System.setProperty("hadoop.home.dir", "D:\\工作\\大数据\\hadoop\\软件\\hadoop-2.6.1")
  System.setProperty("HADOOP_USER_NAME", "hadoop")

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("kmeans").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val hiveContext = new HiveContext(sc)
    import hiveContext.implicits._


    //先从hive中加载到日志数据
    hiveContext.sql("use mymllib")
    val frame: DataFrame = hiveContext.sql("show tables")
    /**
      * +-------------+--------+-----------+
      * |orderlocation|totalqty|totalamount|
      * +-------------+--------+-----------+
      * |        GUIHE|   30446|    6571570|
      * |           ZY|   27049|    6307626|
      * |           TM|    4363|     708261|
      * |           TS|   17961|    2236106|
      * |           TY|   23386|    4330205|
      * |       HUAXIN|    4186|     487099|
      * |           HL|   13426|    1637889|
      * |         BYYZ|    4401|     533953|
      * |           DY|     355|      55195|
      * |       YINZUO|   48336|    8471349|
      * |       TAIHUA|   41002|    9180132|
      * |           RM|   21627|    3010800|
      * |     DOGNGUAN|   33422|    5726613|
      * |           LJ|   47375|    9865094|
      * |         ZHAO|   24996|    4587718|
      * |           LZ|     706|     123817|
      * |           ZM|   30513|    4267355|
      * +-------------+--------+-----------+
      */
    val data: DataFrame = hiveContext.sql("select a.orderlocation, sum(b.itemqty) totalqty,sum(b.itemamout) totalamount from tbl_stock a join tbl_stockdetail b on a.orderid=b.orderid group by a.orderlocation")

    data.show()

    //将hive中查询过来的数据，每一条变成一个向量，整个数据集变成矩阵
    val parserData: RDD[linalg.Vector] = data.map {
      case Row(_, totalqty, totalamount) => {
        val features: Array[Double] = Array[Double](totalqty.toString.toDouble, totalamount.toString.toDouble)
        //  将数组变成机器学习中的向量
        Vectors.dense(features)
      }
    }
    //用kmeans对样本向量进行训练得到模型
    //中心点
    val numCluster = 3
    //最大迭代次数
    val maxIterations = 20
    val model: KMeansModel = KMeans.train(parserData, numCluster, maxIterations)
    val resrdd = data.map {
      case Row(orderLocation, totalQty, totalMount) => {
        val features: Array[Double] = Array[Double](totalQty.toString.toDouble, totalMount.toString.toDouble)
        val lineVector: linalg.Vector = Vectors.dense(features)
        //预测
        val prediction: Int = model.predict(lineVector)
        s"$orderLocation   $totalQty     $totalMount   $prediction"
        (orderLocation, totalQty, totalMount, prediction.toDouble)
      }
    }

    val ressorted: RDD[(Any, Any, Any, Double)] = resrdd.sortBy(_._4)

    /**
      * (TM,4363,708261,1.0)
      * (ZY,27049,6307626,0.0)
      * (TS,17961,2236106,1.0)
      * (TY,23386,4330205,0.0)
      * (HUAXIN,4186,487099,1.0)
      * (RM,21627,3010800,0.0)
      * (HL,13426,1637889,1.0)
      * (DOGNGUAN,33422,5726613,0.0)
      * (BYYZ,4401,533953,1.0)
      * (ZHAO,24996,4587718,0.0)
      * (DY,355,55195,1.0)
      * (ZM,30513,4267355,0.0)
      * (LZ,706,123817,1.0)
      * (YINZUO,48336,8471349,2.0)
      * (TAIHUA,41002,9180132,2.0)
      * (LJ,47375,9865094,2.0)
      */
    ressorted.foreach(println)

    /**
      * GUIHE   30446     6571570   2
      * ZY   27049     6307626   2
      * TM   4363     708261   0
      * TS   17961     2236106   0
      * TY   23386     4330205   2
      * HUAXIN   4186     487099   0
      * HL   13426     1637889   0
      * BYYZ   4401     533953   0
      * DY   355     55195   0
      * YINZUO   48336     8471349   1
      * TAIHUA   41002     9180132   1
      * RM   21627     3010800   0
      * LJ   47375     9865094   1
      * DOGNGUAN   33422     5726613   2
      * ZHAO   24996     4587718   2
      * LZ   706     123817   0
      * ZM   30513     4267355   2
      */
    resrdd.foreach(println)

    sc.stop()

  }


}
