
#!/bin/bash
data_prePath=$1
tmpPath=$2

#build graph (will create CSVDataset folder and save csv files)
python ${WORKSPACE}/src/build_graph.py --data_in ${data_prePath} --gnn_tmp ${tmpPath}
