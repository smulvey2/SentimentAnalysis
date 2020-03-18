import java.util.ArrayList;
import java.util.List;

public class CrossValidation {
    /*
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    public static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
    	
    	double errors;
        double total = 0.0;
        List<List<Instance>> list = new ArrayList<List<Instance>>();
        double[] totals = new double[k];
        
        for (int h = 0; h < k; h++) {
            List<Instance> sublist = new ArrayList<Instance>();
            for (int i = 0; i < trainData.size(); i++) {
                if (i / (trainData.size() / k) == h) {
                    sublist.add(trainData.get(i));
                }
            } 
            list.add(h, sublist);
        }
        for (int i = 0; i < k; i++) {
            clf = new NaiveBayesClassifier();
            errors = 0;
            List<Instance> mini = new ArrayList<Instance>();
            if (i == 0) {
                for (int j = 1; j < k; j++) {
                    for (int l = 0; l < list.get(j).size(); l++) {
                        mini.add(list.get(j).get(l));
                    }
                }
                clf.train(mini, v);
            }
            
            else if (i == k - 1) {
                for (int j = 0; j < k - 1; j++) {
                    for (int l = 0; l < list.get(j).size(); l++) {
                        mini.add(list.get(j).get(l));
                    }
                }
                clf.train(mini, v);
            }
            
            else {
                for (int j = 0; j < i; j++) {
                    for (int l = 0; l < list.get(j).size(); l++) {
                        mini.add(list.get(j).get(l));
                    }
                }
                
                for (int j = i + 1; j < k; j++) {
                    for (int l = 0; l < list.get(j).size(); l++) {
                        mini.add(list.get(j).get(l));
                    }
                }
                clf.train(mini, v);
            }
            
            for (Instance instance : list.get(i)) {
                ClassifyResult classification = clf.classify(instance.words);
                if (classification.label != instance.label) {
                    errors++;
                }
            }
            total = 1 - errors / list.get(i).size();
            totals[i] = total;
        }
        
        double x = 0.0;
        double allTotals = 0.0;
        
        for (int i = 0; i < totals.length; i++) {
            allTotals += totals[i];
            x++;
        }
        
        return allTotals / x;
    }
}
