import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.util.LongAccumulator;

public class Application {

    public static void main(String[] args) {

        System.out.println("******************* Logistic Regression Algorithm :*******************");
        //SparkSession & Load CSV file into Dataset<Row
        SparkSession spark = SparkSession
                .builder()
                .appName("Spring Boot App with Spark SQL")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> carsDF = spark.read()
                .option("header","true")
                .option("treatEmptyValuesAsNulls", "true")
                .option("inferSchema", "true")
                .option("mode","DROPMALFORMED")
                .option("delimiter",";")
                .csv("src/main/resources/cars.csv")
                .select("Cylinders","Displacement","Horsepower","Weight","Acceleration","Model");



        // Assembling Columns into Features
        System.out.println("******************* Assembler :*******************");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(carsDF.columns())
                .setOutputCol("features");

        Dataset<Row> cars = assembler.setHandleInvalid("skip").transform(carsDF);
        cars.foreach((ForeachFunction<Row>) row-> System.out.println(row));
        System.out.println("End : ******************* Assembler :*******************");


        // Labeling our Data with two labels "powerful"(1) ,and "weak"(0)
        System.out.println("******************* Display Data with features and label :*******************");
        Dataset<Row> horsepowerLabel = cars
                .withColumn("label",
                        functions.when(cars.col("Horsepower").lt(100),0) // car has a weak horsepower
                        .otherwise(1)); // car has a powerful horsepower

        horsepowerLabel.show((int) horsepowerLabel.count());
        System.out.println("End : ******************* Display Data with features and label :*******************");


        //Count frequency of each label
        System.out.println("******************* Count frequency of each label:*******************");
        Dataset<Row> countLabel = horsepowerLabel.groupBy("label").count();
        countLabel.foreach((ForeachFunction<Row>) row-> System.out.println("Label:" +row.get(0)+"\t count:"+row.get(1)));
        System.out.println("End : ******************* Count frequency of each label:*******************");


        //Split Dataset into Training and Test
        Dataset<Row>[] splits = horsepowerLabel.randomSplit(new double[] {0.8,0.2}, 12345);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        // Configuration of the algorithm
        LogisticRegression logReg= new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.0)
                .setElasticNetParam(0.8)
                .setFeaturesCol("features")
                .setLabelCol("label");

        // Getting the model of algorithm with our training dataset
        LogisticRegressionModel logRegModel= logReg.fit(training);

        // Evaluating our model with test dataset
        LogisticRegressionSummary logRegSum = logRegModel.evaluate(test);
        System.out.println("******************* Precision & Recall summary of our Model :*******************");
        System.out.println("Precision Summary : "+logRegSum.weightedPrecision());
        System.out.println("Recall Summary : "+logRegSum.weightedRecall());
        System.out.println("End : ******************* Precision & Recall summary of our Model:*******************");


        // Calculating a confusion Matrix
        LongAccumulator tp = spark.sparkContext().longAccumulator();
        LongAccumulator fp = spark.sparkContext().longAccumulator();
        LongAccumulator fn = spark.sparkContext().longAccumulator();
        LongAccumulator tn = spark.sparkContext().longAccumulator();

        // Note : The default value for threshold is 0.5
        test.foreach((ForeachFunction<Row>) row-> {
            Vector features = row.getAs("features");
            double predicat = logRegModel.predict(features);
            switch ((int) row.getAs("label")){
                case 0:
                    if (predicat < 0.5) tn.add(1);
                    else fp.add(1);
                    break;
                case 1:
                    if (predicat < 0.5) fn.add(1);
                    else tp.add(1);
                    break;
                default:
                    System.out.println("Something wrong ");
            }
        });
        System.out.println("******************* Calculating a confusion Matrix *******************");
        System.out.println("Logistic Regression Model --- TP: "+tp.value()+ ", FP: "+ fp.value() +", TN: "
                + tn.value() +", FN: "+fn.value() );
        System.out.println(" 1. The number of correct predictions for each label:");
        System.out.println(" powerful horsepower classified as powerful : TP= "+tp.value());
        System.out.println(" weak horsepower classified as weak : TN= "+tn.value());
        System.out.println(" 2. The number of incorrect predictions for each label:");
        System.out.println(" powerful horsepower classified as weak : FN= "+fn.value());
        System.out.println(" weak horsepower classified as powerful : FP= "+fp.value());
        System.out.println("End : ******************* Calculating a confusion Matrix *******************");

        System.out.println("End : ******************* Logistic Regression Algorithm :*******************");




    }


}
