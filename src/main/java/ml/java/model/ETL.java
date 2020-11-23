package ml.java.model;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.Pipeline;
import scala.Tuple2;

import java.util.function.Predicate;

import java.util.Arrays;
import java.util.stream.Stream;

public class ETL {
    private String pathDataset;
    SparkSession spark = SparkSession.builder().appName("ML Java").getOrCreate();

    public ETL(String path){
        this.pathDataset = path;
    }

    public void run(){
        Dataset<Row> airbnbDF = spark.read().parquet(this.pathDataset);
        airbnbDF.select("neighbourhood_cleansed", "room_type", "bedrooms", "bathrooms",
                "number_of_reviews", "price").show(5);

        final Dataset<Row>[] datasets = airbnbDF.randomSplit(new double[]{.8, .2}, 42);
        Dataset<Row> trainDF = datasets[0];
        Dataset<Row> testDF = datasets[1];

        Predicate<Tuple2<String, String>> dataType = i -> (i._2().equals("StringType"));
        Stream<Tuple2<String, String>> categoricalCols = Arrays.stream(trainDF.dtypes()).filter(dataType);
        categoricalCols.forEach(System.out::println);

        VectorAssembler vecAssembler = new VectorAssembler()
                .setInputCols(new String[]{"bedrooms"})
                .setOutputCol("features");

        Dataset<Row> vecTrainDF = vecAssembler.transform(trainDF);
        vecTrainDF.select("bedrooms", "features", "price").show(10);


        LinearRegression lr = new LinearRegression()
                .setFeaturesCol("features")
                .setLabelCol("price");


        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {vecAssembler, lr});
        PipelineModel pipelineModel = pipeline.fit(trainDF);

        Dataset<Row> predDF = pipelineModel.transform(testDF);
        predDF.select("bedrooms", "features", "price", "prediction").show(10);

    }



    public static void main(String[] args) {
        ETL etl = new ETL("/c/Users/mrugeles/Documents/DataDiscipline/projects/LearningSparkV2/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/");
        etl.run();

    }
}
