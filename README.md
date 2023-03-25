## Natural Langugae Processing 
This repository contains the Assignments done for the course CS 613 : Natural Language Processing course offerd at IIT Gandhinagar during Semster-1 2021-22.    

## Crawling Data

In this assignment data was scrapped from twitter using the <i>twint</i> API. Tweets related to India on the discussing about topics of Pollution, Climate Change, Eco Friendly and Flood were scrapped. 

Word cloud for data for each topic (i.e. Pollution, Climate Change, Eco Friendly and Flood) was produced. The word cloud for pollution is shown.

![alt-txt](https://github.com/devanshuThakar/Natural-Language-Processing/blob/main/Assignment-1/Pollution_wordcloud.png?raw=true)

## Processing and Understanding Data

In this part a statistical analysis of the Data like frequency distribution of words, validiating the language annotation assigned by Twitter, fitting the Data with the <b>Heap's Law</b>.

According to <b>Heap's Law</b>, the size of vocabulary $|V|$ and number of tokens $N$ are related by the following expression : 
$$|V| = K N^{\beta}$$
where $K$ and $\beta$ parameters. The plot is shown below : 

![alt-txt](https://github.com/devanshuThakar/Natural-Language-Processing/blob/main/Assignment-1/Heaps_Law_plot.png?raw=true)