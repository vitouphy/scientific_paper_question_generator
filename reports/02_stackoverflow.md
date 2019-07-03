

## Report 02 - StackOverFlow



### Data Extraction

Previously, we have extracted top tags in ai.stackexchange.com and used it to hand-pick the same (or similar) tag in stackoverflow.

Dataset retrieved from: https://archive.org/details/stackexchange

Published at: June 03, 2019

**Tags**: "neural-network", "conv-neural-network", "recurrent-neural-network", "convolutional-neural-network", "biological-neural-network", "neural-network-tuning", "machine-learning", "machine-learning-model", "deep-learning", "reinforcement-learning", "natural-language-processing", "nlp", "stanford-nlp", "natural-language-process", "image-recognition", "genetic-algorithm", "evolutionary-algorithm", "classification", "game-ai", "text-classification", "computer-vision", "q-learning", "dataset", "backpropagation", "lstm", "object-recognition", "gradient-descent", "unsupervised-learning", "chatbot", "prediction", "monte-carlo-tree-search", "monte-carlo", "artificial-intelligence", "data-science", "robotics", "linear-regression", "non-linear-regression", "generative-adversarial-network", "game-theory", "pattern-recognition", "activation-function", "statistics", "probability", "probability-density", "probability-theory", "word-embedding", "voice-recognition", "bioinformatics", "word2vec", "supervised-learning", "computational-linguistics", "sentiment-analysis", "linear-algebra", "combinatorics", "regression", "logistic-regression", "time-series", "hypothesis-test", "bayesian-networks", "hierarchical-clustering", "clustering", "random-forest", "data-mining", "bigdata", "algorithm", "graph-algorithm", "sorting-algorithm", "algorithm-design", "tensorflow", "keras"



![alt text](./images/xml_data.png)



**Property of XML Dataset**: Here is the XML for the dataset. Each row is a post. 

1. **PostTypeId=1**: Question with title and tags.

2. **PostTypeId=2**: Answer to a question (denoted by ParentId, the question it answers)

   

*****



### Extractive Procedure

Below is the figure for selecting relevants (Q, A) pairs from stackoverflow dataset.

![alt text](./images/data_extractive_process.png)

1. Separate between questions and answers
2. Ignore question that has no answer
3. Group questions into (i) tag-related questions (ii) all-other questions
4. We don't want to select all questions. Instead we pick only K questions. 
   1. Choose all tag-related questions
   2. If that does not fulfill K questions yet, select randomly from non-tag related questions sets until we have K questions.
5. Each question has one or more answers. Merge each answer to its corresponding question to produce a new row. Save them into an XML file
6. Shuffle the **627,904** rows dataset and split into 3 portions following (90-5-5 rules) to produce train, dev and test sets. We then have, **565,114** for training set, **31,395** for dev set, and **31,395** for test set.

![alt text](./images/data_extractive_count.png)

------



### Data Cleaning

The data espeicially the answers can be slightly messy. Not only it has text but also HTML tags, Latex tags and codings. Below is an example.

**What we see:**

![img](https://lh4.googleusercontent.com/RRD703dMKYUyWKvNInTSDTU-tkgm1hs6eIdDqVmqC0Gn-NOH5AFj9plmyAwxNgrWbJ0c9Qwf0hQ6AtGaNFXabaFTboXv-o1Wku4JNeVVxJt20oTvmlhMsS-9unI2PeUzzSCYzrPKFkI)

-------

**Behind the scene:**

```html
<p>Recently I simulated the Gambler's Problem in RL:</p>  <p><a href="https://i.stack.imgur.com/qNMCK.png" rel="nofollow noreferrer"><img src="https://i.stack.imgur.com/qNMCK.png" alt="enter image description here"></a></p>  <p>Now, the problem is, the curve does not at all appear the way as given in the book. The "best policy" curve appears a lot more undulating than it is shown based on the following factors:</p>  <ul> <li>Sensitivity (i.e. the threshold for which you decide the state values have converged).</li> <li>Probability of heads (expected).</li> <li>Depending the value of sensitivity it also depends on whether I find the policy by finding the action (bet) which cause the maximum return by using <span class="math-container">$&gt;$</span> or by using <span class="math-container">$&gt;=$</span> in the following code i.e: </li> </ul>    <pre><code> initialize maximum = -inf  best_action = None  loop over states:     loop over actions of the state:        if(action_reward&gt;maximum):           best_action = action </code></pre>    <p>Also note that if we make the final reward as 101 instead of 100 the curve becomes more uniform. This problem has also been noted in the following thread.</p>  <p>So what is the actual intuitive explanation behind such a behaviour of the solution. Also here is the <a href="http://www.incompleteideas.net/book/gamblers.html" rel="nofollow noreferrer">thread</a> where this problem is discussed.</p> 
```



**Procedures to clean text**:

1. Remove **\<code>\</code>** tags including text in between
2. Remove all **\<html>** tags. Keep its inner text
3. Remove all Latex completely
4. Lowercase all words
5. Use **SpaCy** to tokenize the sentences.



The sanitized text from the example above:

```html
Recently I simulated the Gambler's Problem in RL: Now, the problem is, the curve does not at all appear the way as given in the book. The "best policy" curve appears a lot more undulating than it is shown based on the following factors: Sensitivity (i.e. the threshold for which you decide the state values have converged). Probability of heads (expected). Depending the value of sensitivity it also depends on whether I find the policy by finding the action (bet) which cause the maximum return by using or by using in the following code i.e: Also note that if we make the final reward as 101 instead of 100 the curve becomes more uniform. This problem has also been noted in the following thread. So what is the actual intuitive explanation behind such a behaviour of the solution. Also here is the thread where this problem is discussed.
```



_______



### Data Characteristics

### Project Folder

