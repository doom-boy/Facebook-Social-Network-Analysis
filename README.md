
# Social Network Analysis Dashboard: Facebook Pages

In this project, we will be analyzing the social networks in 8 different catagories of Facebook pages. These catagories include: Government, News Sites, Atheletes, Public Figures, TV Shows, Politians, Artist, Company. Delieverable include a "streamlit" dashboard providing an overview of the data itself, social network analysis, and interactive modules. 


## About The Data

The dataset was found through the Stanford Network Analysis Project (SNAP), a library in development since 2004, launched in 2009, for purposes of research efforts pertaining to network analysis and large-scale data-mining efforts. 

The data we used was found under the title: 

**Graph Embedding with Self Clustering: Facebook**

The dataset itself contains 8 seperate CSV files, each with varying numbers of entries (min: 17,263 max: 819,307). All of these CSV files follow a 2 column approach of {node1, node2} where the existence indicates an edge between "node 1" and "node 2".


A link to the arXiv research paper is provided:

https://arxiv.org/abs/1802.03997

## Setup Instruction

### Prerequisites
- Python 3 installed on system
- Terminal

### Windows
1. Clone this repo
2. Open a terminal & navigate to the cloned repository
3. Create and activate virtual environment with the following commands:  
```python -m venv venv```  
```env\Scripts\activate```
4. Run the following command to install necessary libraries (May take some time):  
```pip install streamlit pandas networkx plotly```
5. In the terminal, run the dashboard with the command:  
```streamlit run dashboard.py```

### Mac
1. Clone this repo
2. Open a terminal & navigate to the cloned repository
3. Create and activate virtual environment with the following commands:  
```python3 -m venv venv```  
```source venv/bin/activate```
4. Run the following command to install necessary libraries (May take some time):  
```pip install streamlit pandas networkx plotly```
5. In the terminal, run the dashboard with the command:  
```streamlit run dashboard.py```