# baseline

this repository is supported by Arivazhagan G. Balasubramanian and this is only reproduction of his work.
For further information, check https://arxiv.org/abs/2404.14121 

this is the machine-learning training code and this code is originally written in tensorflow. I wrote in pytorch but the mechanics is same; both library uses the architechture of Convolutional Neural Networks(CNN).  
And, this process is done via super GPU computer. to request, execute  
```
sbatch submit.sh
```
what is sbatch? -see [link](https://slurm.schedmd.com/sbatch.html)
```
squeue -u geethaa
```
or
```
gbck
```
shows the job progress
lsr

show the training situation -> look at the train.txt
## note
pytorch is not available. Learn how to use Tensorflow.
how can we evaluate the model? -> CNN-predict-i.py?
submit.sh is not the suitable word for it: I think training.sh and eval.sh should be prepared so that anyone can work and measure the efficiency of the model. 
```
projinfo 
```
means other members using priority
shows 
```
scancel 2799829
```
means cancelling request
check the results : 
```
cd .jupyter_plots
```
if there is something lost packages, execute
```
module spider <module name>
```
you can search the module name by 
```
```
scontrol is 
see here if you want to make a new virtual environment https://www.c3se.chalmers.se/documentation/applications/python/#virtual-environments

#warning!
Please ignore 
# ml models
*.h5
*.pt
*.ckpt

# other large files
*.zip
*.tar.gz

# explanations of functions
input_parser: This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
output_parser:This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    
