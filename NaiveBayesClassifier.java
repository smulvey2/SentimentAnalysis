import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.lang.Math;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {
	
	private Map<String, Integer> negativeWords = new HashMap<String, Integer>();
	private Map<String, Integer> positiveWords = new HashMap<String, Integer>();
	private Map<Label, Integer> words = new HashMap<Label, Integer>();
	private Map<Label, Integer> docs = new HashMap<Label, Integer>();
	
	private int x;
	
	/**
 	* Trains the classifier with the provided training data and vocabulary size
 	*/
	@Override
	public void train(List<Instance> trainData, int x) {
		
    	// TODO : Implement
    	// Hint: First, calculate the documents and words counts per label and store them.
    	// Then, for all the words in the documents of each label, count the number of occurrences of each word.
    	// Save these information as you will need them to calculate the log probabilities later.
    	//
    	// e.g.
    	// Assume m_map is the map that stores the occurrences per word for positive documents
    	// m_map.get("catch") should return the number of "catch" es, in the documents labeled positive
    	// m_map.get("asdasd") would return null, when the word has not appeared before.
    	// Use m_map.put(word,1) to put the first count in.
    	// Use m_map.replace(word, count+1) to update the value
    	
    	this.words = getWordsCountPerLabel(trainData);
    	this.docs = getDocumentsCountPerLabel(trainData);
    	this.x = x;

    	for (int i = 0; i < trainData.size(); i++) {  // setting pos and neg word count
        	if (trainData.get(i).label == Label.POSITIVE) {
            	for(int j = 0; j < trainData.get(i).words.size(); j++) {
                	String current = trainData.get(i).words.get(j);
                	if (positiveWords.containsKey(current)) {
                    	positiveWords.replace(current, positiveWords.get(current) + 1);
                	}
                	
                	else {
                    	positiveWords.put(current, 1);
                	}
            	}
        	}
        	
        	else {
            	for (int j = 0; j < trainData.get(i).words.size(); j++) {
                	String currWord = trainData.get(i).words.get(j);
                	if (negativeWords.containsKey(currWord)) {
                    	negativeWords.replace(currWord, negativeWords.get(currWord) + 1);
                	}
                	
                	else {
                    	negativeWords.put(currWord, 1);
                	}
            	}
        	}
    	}
	}

	/*
 	* Counts the number of words for each label
 	*/
	@Override
	public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
		
		Map<Label, Integer> words1 = new HashMap<Label, Integer>();
		int negatives = 0;
    	int positives = 0;
    	
    	for (int i = 0; i < trainData.size(); i++) {
        	if (trainData.get(i).label == Label.POSITIVE) {
            	positives+= trainData.get(i).words.size();
        	}
        	
        	else {
            	negatives += trainData.get(i).words.size();
        	}
    	}
    	
    	words1.put(Label.NEGATIVE, negatives);
    	words1.put(Label.POSITIVE, positives);
    	
    	return words1;
	}


	/*
 	* Counts the total number of documents for each label
 	*/
	@Override
	public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
		
		Map<Label, Integer> documents1 = new HashMap<Label, Integer>();
		int negDoc = 0;
		int posDoc = 0;
    	
    	for (int i = 0; i < trainData.size(); i++) {
        	if (trainData.get(i).label == Label.POSITIVE) {
            	posDoc++;
        	}
        	
        	else {
            	negDoc++;
        	}
    	}
    	
    	documents1.put(Label.NEGATIVE, negDoc);    	
    	documents1.put(Label.POSITIVE, posDoc);

    	return documents1;
	}


	/**
 	* Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
 	*/
	private double p_l(Label label) {
    	// Calculate the probability for the label. No smoothing here.
    	// Just the number of label counts divided by the number of documents.
		
		double negatives1 = this.docs.get(Label.NEGATIVE);
    	double probability = 0;
    	double positives1 = this.docs.get(Label.POSITIVE);
    	
    	
    	if (label == Label.POSITIVE) {
        	probability = positives1 / (positives1 + negatives1);
    	}
    	
    	else {
        	probability = negatives1 / (positives1 + negatives1);
    	}
    	
    	return probability;
	}

	/**
	    * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
	    * P(word|NEGATIVE)
	    */
	    private double p_w_given_l(String word, Label label) {
	        // Calculate the probability with Laplace smoothing for word in class(label)
	        
	    	double num = 0;
	        double total = 0;
	        double probabilityL = 0;
	        
	        if (label == Label.POSITIVE) {
	            total = this.words.get(Label.POSITIVE);
	            if (!positiveWords.containsKey(word)) {
	                num = 1;
	            }
	            
	            else {
	                num = positiveWords.get(word) + 1;
	            }
	            probabilityL = num / (this.x + total);
	        }
	        
	        else {
	            total = this.words.get(Label.NEGATIVE);
	            if (!negativeWords.containsKey(word)) {
	                num = 1;
	            }
	            
	            else {
	                num = negativeWords.get(word) + 1;
	            }
	            probabilityL = num / (this.x + total);
	        }

	        return probabilityL;
	    }

	    /**
	    * Classifies an array of words as either POSITIVE or NEGATIVE.
	    */
	    @Override
	    public ClassifyResult classify(List<String> words) {
	        // TODO : Implement
	        // Sum up the log probabilities for each word in the input data, and the probability of the label
	        // Set the label to the class with larger log probability
	    	
	        ClassifyResult cr = new ClassifyResult();
	        Map<Label, Double> logProb = new HashMap<Label, Double>();

	        double negativeL = 0;
	        double negativeT = 0;
	        double negativeD = this.docs.get(Label.NEGATIVE);
	        
	        double positiveD = this.docs.get(Label.POSITIVE);
	        double positiveL = 0;
	        double positiveT = 0;

	        double sum = this.docs.get(Label.POSITIVE) + this.docs.get(Label.NEGATIVE);
	        double probNegLabel = negativeD / sum;
	        double probPosLabel = positiveD / sum;
	        
	        for (int i = 0; i < words.size(); i++) {
	            negativeL += Math.log(p_w_given_l(words.get(i), Label.NEGATIVE));
	            positiveL += Math.log(p_w_given_l(words.get(i), Label.POSITIVE));
	        }
	        
	        negativeT = Math.log(probNegLabel) + negativeL;	        
	        positiveT = Math.log(probPosLabel) + positiveL;

	        if (positiveT >= negativeT) {
	            cr.label = Label.POSITIVE;
	        }
	        
	        else {
	            cr.label = Label.NEGATIVE;
	        }
	         
	        logProb.put(Label.NEGATIVE, negativeT);
	        logProb.put(Label.POSITIVE, positiveT);

	        cr.logProbPerLabel = logProb;
	        return cr;
	    }
}
