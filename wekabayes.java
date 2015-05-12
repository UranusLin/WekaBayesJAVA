import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;
 
public class wekabayes {

  public static void main(String[] args) throws Exception {
	  BufferedReader traind = new BufferedReader(new FileReader("D:\\put train here.arff"));
	  BufferedReader testd = new BufferedReader(new FileReader("D:\\put test here.arff"));
	  FileWriter fw =new FileWriter("D:put ouput file here.txt");
	  double kk[];
	  Instances train = new Instances(traind);
	  Instances test = new Instances(testd);
	  train.setClassIndex(train.numAttributes()-1);
	  test.setClassIndex(test.numAttributes()-1);
	  traind.close();
	  testd.close();
	  NaiveBayes nb = new NaiveBayes();
	  nb.buildClassifier(train);
	  Evaluation eval = new Evaluation(train);
	  eval.crossValidateModel(nb, train, 10, new Random(1));
	  System.out.println(eval.toSummaryString("\nResult\n==========\n",true));
	  System.out.println(eval.toClassDetailsString());
	  System.out.println(eval.toMatrixString());
	  System.out.println(eval.fMeasure(1)+" "+eval.precision(1)+" "+eval.recall(1));
	  kk=eval.evaluateModel(nb, test);
	  for (int i=0;i<kk.length;i++)
	  {
		  fw.write(kk[i]+"\n");
	  }

	  fw.close();
	  
  }
}
