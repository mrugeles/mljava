package ml.java.model;

import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class FeaturesExperiments {

    SparkSession spark = null;
    Dataset<Row> airbnbDF =null;

    public FeaturesExperiments(SparkSession spark, Dataset<Row> airbnbDF){
        this.spark = spark;
        this.airbnbDF = airbnbDF;
    }

    public void LabelEncoder(){
        StringIndexer labelEncoder = new StringIndexer()
                .setInputCols(new String[]{"neighbourhood_cleansed"})
                .setOutputCols(new String[]{"neighbourhood_index"});

        Dataset encoded = labelEncoder.fit(airbnbDF).transform(airbnbDF);
        encoded.select("neighbourhood_cleansed", "neighbourhood_index").show();
    }

    public void OneHotEncoder(){
        StringIndexer labelEncoder = new StringIndexer()
                .setInputCols(new String[]{"room_type"})
                .setOutputCols(new String[]{"room_type_index"});

        OneHotEncoder oneHotEncoder = new OneHotEncoder()
                .setInputCols(new String[]{"room_type_index"})
                .setOutputCols(new String[]{"room_type_ohe"});

        Dataset<Row> labelEncoded = labelEncoder.fit(airbnbDF).transform(airbnbDF);
        labelEncoded.select("room_type", "room_type_index").show(5);

        Dataset<Row> ohEncoded = oneHotEncoder.fit(labelEncoded).transform(labelEncoded);
        ohEncoded.select("room_type", "room_type_index", "room_type_ohe").show(5);
    }

     public static void main(String[] args) {
         String pathDataset = "C:\\Users\\mrugeles\\Documents\\DataDiscipline\\projects\\LearningSparkV2\\databricks-datasets\\learning-spark-v2\\sf-airbnb\\sf-airbnb-clean.parquet\\";
         SparkSession spark = SparkSession
                 .builder()
                 .appName("FeaturesExperiments")
                 .master("local")
                 .config("spark.driver.memory", "550M")
                 .getOrCreate();

         Dataset<Row> airbnbDF = spark.read().parquet(pathDataset);

         FeaturesExperiments exp = new FeaturesExperiments(spark, airbnbDF);
         //exp.LabelEncoder();
         exp.OneHotEncoder();

    }
}
