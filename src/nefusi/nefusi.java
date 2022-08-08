package nefusi;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import org.moeaframework.Executor;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import org.moeaframework.problem.AbstractProblem;

import net.sourceforge.jFuzzyLogic.FIS;

public class nefusi {

public static class nefu extends AbstractProblem {
		
		private static ArrayList<Double[]> data_training;
		private static ArrayList<Double[]> data_test;
		
		static double [] source_test;
		static double [] target_test;
		static double [] source_training;
		static double [] target_training;
		
	    private static String [] valueS = {"null", "poor", "good", "excellent"};
	    private static String [] valueAS = {"COG", "MM", "LM", "RM"};
		
		static String [] v = null;
		
		int i = 0;

		/**
		 * Constructs a new instance of the program
		 */
		public nefu() {
			super(62, 3);
			
	        try {
	            data_training = new ArrayList<Double[]>();
	            final FileInputStream stream = new FileInputStream("datasets/mc-training.txt");
	            @SuppressWarnings("resource")
				final BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
	            String data = "";
	            do {
	                data = reader.readLine();
	                if (data == null || data.trim().length() == 0) {
	                	continue;
	                } else {
	                	final String [] line = data.split(",");
	                    if (line.length == 5) {
	                        try {
	                            final double a = Double.parseDouble(line[0]);
	                            final double b = Double.parseDouble(line[1]);
	                            final double c = Double.parseDouble(line[2]);
	                            final double d = Double.parseDouble(line[3]);
	                            final double e = Double.parseDouble(line[4]);
	                            final Double [] f = { a, b, c, d, e };
	                            data_training.add(f);
	                            //System.out.println(a);
	                        } catch(final NumberFormatException e) {
	                            System.out.println(e);
	                        }
	                    }
	                } // End of the if //
	            } while(data != null);            
	            //System.out.println("INFO: data loaded : " + filename);
	            
	            source_training = new double [data_training.size()];
	            
		        for (int i = 0; i < data_training.size(); i++)
		        	source_training[i] = Double.valueOf(data_training.get(i)[0]);
	            
	        } catch(Exception e) {
	            e.printStackTrace();
	        } 
	        
	        
	        try {
	            data_test = new ArrayList<Double[]>();
	            final FileInputStream stream = new FileInputStream("datasets/geresid-validation.txt");
	            @SuppressWarnings("resource")
				final BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
	            String data = "";
	            do {
	                data = reader.readLine();
	                if (data == null || data.trim().length() == 0) {
	                	continue;
	                } else {
	                	final String [] line = data.split(",");
	                    if (line.length == 5) {
	                        try {
	                            final double a = Double.parseDouble(line[0]);
	                            final double b = Double.parseDouble(line[1]);
	                            final double c = Double.parseDouble(line[2]);
	                            final double d = Double.parseDouble(line[3]);
	                            final double e = Double.parseDouble(line[4]);
	                            final Double [] f = { a, b, c, d, e };
	                            data_test.add(f);
	                            //System.out.println(a);
	                        } catch(final NumberFormatException e) {
	                            System.out.println(e);
	                        }
	                    }
	                } // End of the if //
	            } while(data != null);            
	            //System.out.println("INFO: data loaded : " + filename);
	            
	            source_test = new double [data_test.size()];
	            
		        for (int i = 0; i < data_test.size(); i++)
		        	source_test[i] = Double.valueOf(data_test.get(i)[0]);
	            
	        } catch(Exception e) {
	            e.printStackTrace();
	        }
			
		}
		
		
		/** 
		 * Compute the Pearson Correlation Coefficient
		 */ 
		public static double getPearson(double[] scores1, double[] scores2) {
			double result = 0;
			double sum_sq_x = 0;
			double sum_sq_y = 0;
			double sum_coproduct = 0;
			double mean_x = scores1[0];
			double mean_y = scores2[0];
			for (int i = 2; i < scores1.length + 1; i += 1) {
				double sweep = Double.valueOf(i - 1) / i;
				double delta_x = scores1[i - 1] - mean_x;
				double delta_y = scores2[i - 1] - mean_y;
				sum_sq_x += delta_x * delta_x * sweep;
				sum_sq_y += delta_y * delta_y * sweep;
				sum_coproduct += delta_x * delta_y * sweep;
				mean_x += delta_x / i;
				mean_y += delta_y / i;
			}
			double pop_sd_x = (double) Math.sqrt(sum_sq_x / scores1.length);
			double pop_sd_y = (double) Math.sqrt(sum_sq_y / scores1.length);
			double cov_x_y = sum_coproduct / scores1.length;
			result = cov_x_y / (pop_sd_x * pop_sd_y);
			
			if (Double.isNaN(result))
				return -9;
			else
				return result;
		}
		
		@SuppressWarnings("null")
		/** 
		 * Compute the Spearman Rank Correlation Coefficient
		 */ 
		private static double getSpearman(double [] X, double [] Y) {
		       
			try {
			/* Error check */
		        if (X == null || Y == null || X.length != Y.length) {
		            return (Double) null;
		        }
		        
		        /* Create Rank arrays */
		        int [] rankX = getRanks(X);
		        int [] rankY = getRanks(Y);

		        /* Apply Spearman's formula */
		        int n = X.length;
		        double numerator = 0;
		        for (int i = 0; i < n; i++) {
		            numerator += Math.pow((rankX[i] - rankY[i]), 2);
		        }
		        numerator *= 6;
		        return 1 - numerator / (n * ((n * n) - 1));
			}
			catch (Exception e) {
				return -999;
			}
		    }
		    
		    /* Returns a new (parallel) array of ranks. Assumes unique array values */
		    public static int[] getRanks(double [] array) {
		        int n = array.length;
		        
		        /* Create Pair[] and sort by values */
		        Pair [] pair = new Pair[n];
		        for (int i = 0; i < n; i++) {
		            pair[i] = new Pair(i, array[i]);
		        }
		        Arrays.sort(pair, new PairValueComparator());

		        /* Create and return ranks[] */
		        int [] ranks = new int[n];
		        int rank = 1;
		        for (Pair p : pair) {
		            ranks[p.index] = rank++;
		        }
		        return ranks;
		    }
		
		/** 
		* Construct the fuzzy program
		*/ 
		public static void replace(String [] target){  
			    try {  
			        FileReader fr = new FileReader("fcl/tipperB.fcl");  
			        BufferedReader br = new BufferedReader(fr);  
			        FileWriter fw = new FileWriter("fcl/tipper99.fcl");  
			        PrintWriter bw = new PrintWriter(fw);  
			        String line = null;  
			  
			        while((line=br.readLine()) != null) {  
			        	line = line.replaceAll("aaa", String.valueOf(target[0]));    
			        	line = line.replaceAll("bbb", String.valueOf(target[1])); 
			        	line = line.replaceAll("ccc", String.valueOf(target[2])); 
			        	line = line.replaceAll("ddd", String.valueOf(target[3])); 
			        	line = line.replaceAll("eee", String.valueOf(target[4])); 
			        	line = line.replaceAll("fff", String.valueOf(target[5])); 
			        	line = line.replaceAll("ggg", String.valueOf(target[6])); 
			        	line = line.replaceAll("hhh", String.valueOf(target[7])); 
			        	line = line.replaceAll("iii", String.valueOf(target[8])); 
			        	line = line.replaceAll("jjj", String.valueOf(target[9])); 
			        	line = line.replaceAll("kkk", String.valueOf(target[10]));
			        	line = line.replaceAll("lll", String.valueOf(target[11])); 
			        	line = line.replaceAll("mmm", String.valueOf(target[12])); 
			        	line = line.replaceAll("nnn", String.valueOf(target[13])); 
			        	line = line.replaceAll("ooo", String.valueOf(target[14])); 
			        	line = line.replaceAll("ppp", String.valueOf(target[15])); 
			        	line = line.replaceAll("qqq", String.valueOf(target[16])); 
			        	line = line.replaceAll("rrr", String.valueOf(target[17])); 
			        	
			        	line = line.replaceAll("2aa2", String.valueOf(target[18]));    
			        	line = line.replaceAll("2bb2", String.valueOf(target[19])); 
			        	line = line.replaceAll("2cc2", String.valueOf(target[20])); 
			        	line = line.replaceAll("2dd2", String.valueOf(target[21])); 
			        	line = line.replaceAll("2ee2", String.valueOf(target[22])); 
			        	line = line.replaceAll("2ff2", String.valueOf(target[23])); 
			        	line = line.replaceAll("2gg2", String.valueOf(target[24])); 
			        	line = line.replaceAll("2hh2", String.valueOf(target[25])); 
			        	line = line.replaceAll("2ii2", String.valueOf(target[26])); 
			        	line = line.replaceAll("2jj2", String.valueOf(target[27])); 
			        	line = line.replaceAll("2kk2", String.valueOf(target[28]));
			        	line = line.replaceAll("2ll2", String.valueOf(target[29])); 
			        	line = line.replaceAll("2mm2", String.valueOf(target[30])); 
			        	line = line.replaceAll("2nn2", String.valueOf(target[31])); 
			        	line = line.replaceAll("2oo2", String.valueOf(target[32])); 
			        	line = line.replaceAll("2pp2", String.valueOf(target[33])); 
			        	line = line.replaceAll("2qq2", String.valueOf(target[34])); 
			        	line = line.replaceAll("2rr2", String.valueOf(target[35])); 
			        	
			        	line = line.replaceAll("3aa3", String.valueOf(target[36]));    
			        	line = line.replaceAll("3bb3", String.valueOf(target[37])); 
			        	line = line.replaceAll("3cc3", String.valueOf(target[38])); 
			        	line = line.replaceAll("3dd3", String.valueOf(target[39])); 
			        	line = line.replaceAll("3ee3", String.valueOf(target[40])); 
			        	line = line.replaceAll("3ff3", String.valueOf(target[41])); 
			        	line = line.replaceAll("3gg3", String.valueOf(target[42])); 
			        	line = line.replaceAll("3hh3", String.valueOf(target[43])); 
			        	line = line.replaceAll("3ii3", String.valueOf(target[44])); 
			        	line = line.replaceAll("3jj3", String.valueOf(target[45])); 
			        	line = line.replaceAll("3kk3", String.valueOf(target[46]));
			        	line = line.replaceAll("3ll3", String.valueOf(target[47])); 
			        	line = line.replaceAll("3mm3", String.valueOf(target[48])); 
			        	line = line.replaceAll("3nn3", String.valueOf(target[49])); 
			        	line = line.replaceAll("3oo3", String.valueOf(target[50])); 
			        	line = line.replaceAll("3pp3", String.valueOf(target[51])); 
			        	line = line.replaceAll("3qq3", String.valueOf(target[52])); 
			        	line = line.replaceAll("3rr3", String.valueOf(target[53])); 
			        	
			        	line = line.replaceAll("4aa4", String.valueOf(target[54]));    
			        	line = line.replaceAll("4bb4", String.valueOf(target[55])); 
			        	line = line.replaceAll("4cc4", String.valueOf(target[56])); 
			        	line = line.replaceAll("4dd4", String.valueOf(target[57])); 
			        	line = line.replaceAll("4ee4", String.valueOf(target[58])); 
			        	line = line.replaceAll("4ff4", String.valueOf(target[59])); 
			        	line = line.replaceAll("4gg4", String.valueOf(target[60])); 
			        	line = line.replaceAll("4hh4", String.valueOf(target[61])); 
			        	
			        
			        	bw.println(line);  
			            } 
			        
			        fr.close();
			        fw.close();
			        
			       

			        
			        }  
			    catch(Exception e){e.printStackTrace();} 
		   }
		   
		/**
		 * Copies the FLC
		 */
		private static void copyFileUsingStream(String s, String d) throws IOException {
			    
	        File inputFile = new File(s);
	        File tempFile = new File("fcl/temp");

	        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
	        BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile));

	        String currentLine;

	        while((currentLine = reader.readLine()) != null) {
	            if(currentLine.contains("null") == false)
	            	writer.write(currentLine + System.getProperty("line.separator"));
	        }
	        
	        writer.close(); 
	        reader.close(); 
	        //tempFile.renameTo(inputFile);
			
			
			File source = new File ("fcl/temp");
			File dest = new File (d);
				
			InputStream is = null;
		    OutputStream os = null;
		    try {
		        is = new FileInputStream(source);
		        os = new FileOutputStream(dest);
		        byte[] buffer = new byte[1024];
		        int length;
		        while ((length = is.read(buffer)) > 0) {
		            os.write(buffer, 0, length);
		        }
		    } finally {
		        is.close();
		        os.close();
		    }
		}
		
		/**
		 * Calculates Validation
		 */
		public static void printValidation () {
			   	
			FIS fis = new FIS ();
	        try {
	        	fis = FIS.load("fcl/temp", true);
			} catch (Exception e) {
				return;
			}
			
	       
	        target_test = new double [data_test.size()];
	        for (int i = 0; i < data_test.size(); i++) {
	        	fis.setVariable("alga", data_test.get(i)[1]);
	        	fis.setVariable("algb", data_test.get(i)[2]);
	        	fis.setVariable("algc", data_test.get(i)[3]);
	        	fis.setVariable("algd", data_test.get(i)[4]);
	        
	        	// Evaluate
	        	fis.evaluate();

	        	// Training output 
	        	target_test[i] = fis.getVariable("score").defuzzify();
	        
	        }

	        
	    	PrintWriter pw = null;
			try {
				pw = new PrintWriter(new FileWriter("out.txt"));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	    	 
	    		pw.write((int) getPearson(source_test, target_test));
	     
	    	pw.close();
			
		}
		

		/**
		 * Constructs a new solution and defines the bounds of the decision
		 * variables.
		 */
		@Override
		public Solution newSolution() {
			Solution solution = new Solution(getNumberOfVariables(), 
					getNumberOfObjectives());

			for (int i = 0; i < getNumberOfVariables(); i++) {
				solution.setVariable(i, new RealVariable(0, 3.99));
			}

			return solution;
		}
		

		
		/**
		 * Evaluation function
		 */
		@Override
		public void evaluate(Solution solution) {
			double[] x = EncodingUtils.getReal(solution);
			double[] f = new double[numberOfObjectives];
			
			i++;
			
			
	        String[] geneString = new String[62];
	        for (int i = 0; i < 54; i++) {
	        	int num = (int) x[i];
	        	geneString[i] = valueS[num];
	        	if (i == 12)
	        		geneString[i] = valueAS[num];
	        }
	        
	        
	        for (int i = 54; i < 62; i++) {      	
	        	double val = (double)Math.round(x[i]/2.99 * 1000d) / 1000d;
	        	val = val * 0.15;
	        	
	 		    double a = Double.valueOf(val);
	 		    double b = Double.valueOf(val + a);
	 		    
	 		    double c = Double.valueOf(val * 2);
	 		    double d = Double.valueOf(val + c);
	 		    double e = Double.valueOf(val + d);
	 		    double f1 = Double.valueOf( val + e);
	 		    
	 		    double g = Double.valueOf(val * 4);
	 		    double h = Double.valueOf(val + g);
	 		    
	 		    
	 		   geneString[54] = String.valueOf((double)Math.round(a * 1000d) / 1000d);
	 		   geneString[55] = String.valueOf((double)Math.round(b * 1000d) / 1000d);
			    
	 		   geneString[56] = String.valueOf((double)Math.round(c * 1000d) / 1000d);
	 		   geneString[57] = String.valueOf((double)Math.round(d * 1000d) / 1000d);
	 		   geneString[58] = String.valueOf((double)Math.round(e * 1000d) / 1000d);
	 		   geneString[59] = String.valueOf((double)Math.round(f1 * 1000d) / 1000d);
			    
	 		   geneString[60] = String.valueOf((double)Math.round(g * 1000d) / 1000d);
	 		   geneString[61] = String.valueOf((double)Math.round(h * 1000d) / 1000d);
	        
	        }
	        
	        
	        String fileName = "fcl/result.fcl";
			replace (geneString);								 									   	
			FIS fis = new FIS ();
	        try {
	        	copyFileUsingStream("fcl/tipper99.fcl", fileName);
	        	fis = FIS.load(fileName, true);
			} catch (Exception e) {
				return;
			}

	        
	        target_training = new double [data_training.size()];
	        for (int i = 0; i < data_training.size(); i++) {
	        	fis.setVariable("alga", data_training.get(i)[1]);
	        	fis.setVariable("algb", data_training.get(i)[2]);
	        	fis.setVariable("algc", data_training.get(i)[3]);
	        	fis.setVariable("algd", data_training.get(i)[4]);
	        
	        // Evaluate
	        fis.evaluate();

	        // Training output 
	        target_training[i] = fis.getVariable("score").defuzzify();
	        
	        }

	        f[0] = -getPearson(source_training, target_training);
	        
	        target_test = new double [data_test.size()];
	        for (int i = 0; i < data_test .size(); i++) {
	        	
	        	fis.setVariable("alga", data_test.get(i)[1]);
	        	fis.setVariable("algb", data_test.get(i)[2]);
	        	fis.setVariable("algc", data_test.get(i)[3]);
	        	fis.setVariable("algd", data_test.get(i)[4]);
	        
	        // Evaluate
	        fis.evaluate();

	        // Output
	        target_test[i] = fis.getVariable("score").defuzzify();
	        
	        }

	        f[1] = -getPearson(source_test, target_test);
	        
	        // Rules -uncomment when it is necessary to know the number of rules
	        /*int acum = 0;
	        for (int i = 0; i < 54; i=i+3) {
	        	int num = (int) x[i];
	        	if (num == 0)
	        		acum++;
	        }
	        
	        f[2] = 18-acum;*/
	        
	        solution.setObjectives(f);
	        
	        
		}
		
		
		
	}


	//main
	public static void main(String[] args) {
		//configure and run the main function
		
		long startTime = System.nanoTime();
		
		NondominatedPopulation result = new Executor()
				.withProblemClass(nefu.class)
				.withAlgorithm("DE")
				.withMaxEvaluations(10000)
				.run();
		 
		long endTime = System.nanoTime();
		long timeElapsed = endTime - startTime;
        System.out.println("Execution time in milliseconds : " +
                timeElapsed / 1000000);

		System.out.format("Training	Test \n");
		
		for (Solution solution : result) {
			System.out.format("%.4f		%.4f \n",
					-1 * solution.getObjective(0),
					-1 * solution.getObjective(1));
		}
			
		
	}
	
}

