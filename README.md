# nefusi
 Source code from the NEFUSI project
 
 Please note that this is an intermediate version of the code.
 
 It is recommended to import the project with an IDE such as Eclipse. When this is not possible, you can proceed manually as follows:
 
 ## Compilation
 ``` ...\nefusi\src>javac -cp combined.jar nefusi/*.java```
 
 ## Execution
 ``` ...\nefusi\src>java -cp .;combined.jar nefusi.nefusi```
 
 Then, you will see something like this:
 ```
 Execution time in milliseconds : 277719
 Training        Test
 0.8611          0.6805
 ```
 
 What means that the neurofuzzy model completed the training phase with 86.11% accuracy, and the test phase with 68.05%.
 Please note that, for example, for this GeReSiD dataset, 68.05% accuracy is closed to state-of-the-art.
 
 (Please be aware that the running the software under current configuration takes more than 5 minutes (11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz   1.80 GHz))
 
  ## Related papers
 - Jorge Martinez-Gil, Riad Mokadem, Josef Küng, Abdelkader Hameurlain: A Novel Neurofuzzy Approach for Semantic Similarity Measurement. DaWaK 2021: 192-203
 - Jorge Martinez-Gil, Jose Manuel Chaves-Gonzalez: Sustainable Semantic Similarity Asssessment. J. Intell. Fuzzy Syst. 43(5): 6163-6174 (2022)
 
  ## Funding
  The development of NEFUSI is funded in the project NGI Zero Discovery by the NLnet Foundation and the European Commission. Project number: 2021-04-069
 
  ## License
  MIT
