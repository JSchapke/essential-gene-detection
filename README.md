# Essential gene detection with Graph Neural Networks
Contains code for: EPGAT: Gene Essentiality Prediction With Graph Attention Networks,  
by João Schapke, Anderson Tavares,  Mariana Recamonde-Mendoza (https://doi.org/10.1109/tcbb.2021.3054738).
  
Data used is available at: https://drive.google.com/file/d/18TyM7WvZe5QxGCEAxJUWH0kHmi2fWCxa/view?usp=sharing

  
To train GAT in a dataset run:  
```python runners/run_gat.py <OPTIONS>```  
  
To see which options are available:  
```python runners/run_gat.py --help```  
  
Example:  
```python runners/run_gat.py --train --organism human --ppi string --sublocs --expression```  
This will train and evaluate on the human genome dataset with additional data of gene expression profiles and subcellular localizations information.
