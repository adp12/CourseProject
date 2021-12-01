# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

Team AHR

Members

●	Anthony Petrotte (adp12@illinois.edu)

●	Hrishikesh Deshmukh (hcd3@illinois.edu)

●	Rahul Jonnalagadda (rjonna2@illinois.edu)



11/24 To Dos


	• Solidify relevance for Pruning / data to be stored
		○ Ranks based on similar queries to consider
			§ Creates new challenges for accurately reducing 
			§ Structured Query Expansion
		○ Query expansion will allow for weighted additions to the query (i.e. doc tags)
		○ The BM25F will allow for field related weighting (could be used for final weighting after docs and subdocs are pruned to include title in relevance weighting)
	• Solidify Metric Calculation
		○ How to effectively combine the relevance scores with the sentiment scores for each weighted subdoc
	• How to store datasets
		○ Currently thinking locally organized text files for ease/speed of implementation
	• How to present and provide user interface
		○ Currently thinking plotly dash library
    		○ Will require a little extra work as there is an interface to learn
	• Clean up repo for submission
		○ Creating documentation
		○ Creating use guide
		○ Abstracting config file


11/30 To Ds


	• Broadly:
		○ How to use
		○ Method of data storage
		○ Method of data presentation
	• More specific:
		○ Metric calculation
			§ Issues may be arising with the sentiment from the transformer
			§ Hugging face provides a method to add to the training of a model to make it work more specifically to a task
			§ If we can create a labeled data set with some examples, we may be able to improve it
			§ Not as bad as it sounds
		○ Dataset storage
			§ Still thinking easiest method is local text files for convenience
		○ Making it use-able
			§ This is where a dash board would come in handy
			§ If everything is tuned and working, you could set it to create a few sets for a list of ticker symbols
			§ The dashboard would then be used to present the sets of data

