# Decision Tree
# Heuristics: Information Gain, Majority Error, Gini Index
<pre>
1. Execute with: 
	
	./run.sh		# Runs all three required instances with all required depths.  
	
	---For Individual Runs---  
	./run_car.txt 		#  Runs the car dataset, depths 1-6.  
	./run_bank.txt		# Runs the bank dataset with numeric median, depths 1-16.  
	./run_bank_unkn.txt	# Runs the bank dataset with median and replacement of 'unkown'values, depths 1-16.  

NOTE: Output contains both test and training errors. 

2. To run with other data sets the format is as follows:  

	python3 ID3.py &lt;trainfile> &lt;testfile> &lt;depth> &lt;flag> 

Flag Options:  
	No flag runs with each numeric attribute treated as a distinct value.   
	'-num'   :  For numeric values to split based on median.  
	'-unkn'  :  Same as '-num' but with replacement of 'unkown' values.  

When run, three trees are constructed corresponding to each heuristic respectively.  
</pre>
