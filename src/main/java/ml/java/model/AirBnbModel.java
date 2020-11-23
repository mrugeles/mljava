package ml.java.model;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.function.Predicate;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class AirBnbModel<T> {

    SparkSession spark = null;
    Dataset<Row> airbnbDF =null;
    Dataset<Row> trainDF = null;
    Dataset<Row> testDF = null;
    private PipelineModel baselineModel = null;
    private PipelineModel model = null;
    private T learner = null;

    public PipelineModel getBaselineModel() {
        return baselineModel;
    }

    public PipelineModel getModel() {
        return model;
    }

    public AirBnbModel(SparkSession spark, T learner, String pathDataset, float testSize){
        this.spark = spark;
        this.airbnbDF = spark.read().parquet(pathDataset);
        this.learner = learner;

        float trainSize = 1 - testSize;
        final Dataset<Row>[] datasets = airbnbDF.randomSplit(new double[]{trainSize, testSize}, 42);
        this.trainDF = datasets[0];
        this.testDF = datasets[1];

    }


    public String getLearnerClass(PipelineModel pipelineModel){
        Predicate<Transformer> findLearner = t -> (t.getClass().getName().contains("org.apache.spark.ml.regression"));

        Transformer[] learner = Arrays.stream(pipelineModel.stages()).filter(findLearner).toArray(Transformer[]::new);
        if (Arrays.stream(learner).findFirst().isPresent()) {
            String fullClassName = Arrays.stream(learner).findFirst().get().getClass().getName();
            String[] classPackage = fullClassName.split("\\.");
            return classPackage[classPackage.length - 1];
        }

        return null;
    }
    public StringIndexer buildLabelEncoder(String[] features) {
        String[] columnsIndexes = Arrays.stream(features).map(s -> s.concat("_index")).toArray(String[]::new);

        return new StringIndexer()
                .setInputCols(features)
                .setOutputCols(columnsIndexes)
                .setHandleInvalid("skip");
    }

    public OneHotEncoder buildOneHotEncoder(String[] features) {

        String[] columnsIndexes = Arrays.stream(features).map(s -> s + "_index").toArray(String[]::new);
        String[] oheColumns = Arrays.stream(features).map(s -> s + "_ohe").toArray(String[]::new);

        StringIndexer labelEncoder = new StringIndexer()
                .setInputCols(features)
                .setOutputCols(columnsIndexes);

        return new OneHotEncoder()
                .setInputCols(columnsIndexes)
                .setOutputCols(oheColumns);
    }


    public void trainBaseLineModel(String[] features) {

        VectorAssembler vecAssembler = new VectorAssembler()
                .setInputCols(features)
                .setOutputCol("features");

        LinearRegression learner = new LinearRegression()
                .setFeaturesCol("features")
                .setLabelCol("price");

        baselineModel = new Pipeline()
                .setStages(
                        new PipelineStage[]{
                                vecAssembler,
                                learner})
                .fit(trainDF);
    }


    public void trainModel(
            Dataset<Row> trainDF,
            String[] features,
            String [] elFeatures,
            String[] oheFeatures){


        StringIndexer labelEncoder = buildLabelEncoder(elFeatures);
        OneHotEncoder oneHotEncoder = buildOneHotEncoder(oheFeatures);


        VectorAssembler vecAssembler = new VectorAssembler()
                .setInputCols(features)
                .setOutputCol("features");

        model = new Pipeline()
                .setStages(
                        new PipelineStage[]{
                                labelEncoder,
                                oneHotEncoder,
                                vecAssembler,
                                (Regressor)learner
                        }
                ).fit(trainDF);
    }


    public double evalModel(PipelineModel m)  {
        Dataset<Row> predictions = m
                .transform(testDF);

        RegressionEvaluator regressionEvaluator = new RegressionEvaluator()
                .setPredictionCol("prediction")
                .setLabelCol("price")
                .setMetricName("rmse");
        return regressionEvaluator.evaluate(predictions);
    }


    public PipelineModel buildBaselineModel(){
        String[] baseLineFeatures = new String[]{"bedrooms"};

        trainBaseLineModel(baseLineFeatures);
        return this.getBaselineModel();

    }

    public PipelineModel buildModel(){
        String[] labelFeatures = new String[]{
                "host_is_superhost",
                "cancellation_policy",
                "instant_bookable",
                "neighbourhood_cleansed",
                "property_type",
                "room_type"
        };

        String[] oheFeatures = new String[]{
            "host_is_superhost",
                    "room_type",
                    "instant_bookable",
                    "property_type",
                    "cancellation_policy"
        };

        String[] features = new String[]{
                "latitude",
                "longitude",
                "accommodates",
                "bathrooms",
                "bedrooms",
                "beds",
                "host_total_listings_count",
                "cancellation_policy_index",
                "neighbourhood_cleansed_index",
                "property_type_ohe",
                "room_type_ohe",
                "instant_bookable_ohe",
                "host_is_superhost_ohe"
        };

        trainModel(trainDF, features, labelFeatures, oheFeatures);

        return this.getModel();
    }


    public static void main(String[] args){
        Logger logger = LogManager.getLogger(Logger.class.getName());

        float testSize = Float.parseFloat(args[1]);
        String pathDataset = args[0];
        SparkSession spark = SparkSession
                .builder()
                .appName("AirbnbModel")
                .master("local")
                .config("spark.driver.memory", "550M")
                .getOrCreate();

        RandomForestRegressor learner = new RandomForestRegressor()
                .setFeaturesCol("features")
                .setLabelCol("price")
                .setMaxBins(42);
        AirBnbModel<RandomForestRegressor> modelPipeline = new AirBnbModel(spark, learner, pathDataset, testSize);

        PipelineModel baseLineModel = modelPipeline.buildBaselineModel();
        PipelineModel model = modelPipeline.buildModel();

        double baselineScore = modelPipeline.evalModel(baseLineModel);
        double score = modelPipeline.evalModel(model);
        String baselineModelName = modelPipeline.getLearnerClass(modelPipeline.getBaselineModel());
        String modelName = modelPipeline.getLearnerClass(modelPipeline.getModel());

        logger.info(String.format("%f,%s,%f,%s,%f", testSize, baselineModelName, baselineScore, modelName, score));
    }

}
