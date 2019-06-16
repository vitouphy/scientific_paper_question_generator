//
//import java.io.IOException;
//import java.nio.file.Files;
//import java.nio.file.Paths;
//import java.util.*;
//import java.util.concurrent.ConcurrentHashMap;
//import java.util.stream.Collectors;
//import java.util.stream.IntStream;
//import java.util.stream.Stream;
//
//public class Merger {
//
//    static Set tags;
//    static ConcurrentHashMap<String, String[]> csQuestions;
//    static ConcurrentHashMap<String, String[]> randomQuestions;
//    static List<String> data;
//
//    static int DATA_COUNT = 3000000;
//
//
//    public static void main(String[] args) {
//
////        String questionFile = "/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/analysis_001/data/ai.stackexchange.com_questions.csv";
////        String answerFile = "/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/analysis_001/data/ai.stackexchange.com_answers.csv";
////        String outputFile = "/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/analysis_001/data/ai.stackexchange.com.csv";
//
////        String questionFile = args[0];
////        String answerFile = args[1];
////        String outputFile = args[2];
//
//        data = new ArrayList<String>();
//        data.add("QuestionId,AnswerId,Title,Tags,QuestionBody,QuestionScore,AnswerBody,AnswerScore");
//
//        csQuestions = new ConcurrentHashMap<String, String[]>();
//        randomQuestions = new ConcurrentHashMap<String, String[]>();
//        prepareTags();
//        long startTime = System.nanoTime();
//
//        try {
//
//            // Read Question File
//            Stream<String> stream = Files.lines(Paths.get(questionFile)).skip(1);
//            stream.parallel().forEach(Merger::process);
//            Merger.samplingRandomQuestion();
//            System.out.println("number of questions: " + csQuestions.size());
//            // Read Answer and Merge
//            Stream<String> answers = Merger.readAndMergeAnswer(answerFile);
//            Files.write(Paths.get(outputFile), (Iterable<String>) answers::iterator);
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        long endTime = System.nanoTime();
//        System.out.println("It took " + ((endTime - startTime) * 1e-9) + " seconds");
//
//
//    }
//
//    public static void prepareTags() {
//        String[] TAGS = {
//                "local-search","bipartite-matching","swarm-intelligence","hazard","search","long-short-term-memory","graphics","heuristics","splines","lognormal","pymc","estimators","moving-average","social-networks","program-optimization","multinomial","self-replication","closure-properties","computation-tree-logic","branch-and-bound","distributions","russell-norvig","maximum-likelihood","bivariate","amortized-analysis","lower-bounds","code-generation","linear","branching-factors","artificial-consciousness","voice-recognition","sentience","uncountability","out-of-sample","game-theory","generative-model","roc","mediation","weighted-mean","quantiles","soft-question","mdp","r","features","censoring","monte-carlo","multi-tasking","evaluation-strategies","distance-functions","derivative","separation","applications","lavaan","notation","legal","queues","strong-ai","algorithm-analysis","normality-assumption","degrees-of-freedom","pattern-recognition","alphazero","word-embeddings","art-aesthetics","profession","friendly-ai","strings","probabilistic","dependent-types","regression-coefficients","gradient-descent","partitions","hash","model","independence","rewards","arma","cognitive-science","predictor","ab-test","naive-bayes","weighted-regression","applied-theory","scheduling","minimum-spanning-tree","seasonality","geometric-deep-learning","operating-systems","paging","left-recursion","one-way-functions","intercept","causality","numerical-analysis","unification","density-estimation","floating-point","stochastic-processes","finance","mse","parallel-computing","time-series","genetic-algorithms","matching","resource-request","probability-distribution","joint-distribution","genetics","gamma-distribution","normalization","multiple-imputation","epidemiology","lists","kruskal-wallis","memory","teaching-concepts","logistic","image-processing","negative-binomial","measurement-error","space-analysis","academia","hashing","average","sense","bloom-filters","relu","statsmodels","p-vs-np","logarithm","blockchain","imperfect-information","new-ai","divide-and-conquer","confidence-interval","traveling-salesman","neurons","routing","weibull","bagging","learning-theory","rice-theorem","correctness-proof","feedforward","context-sensitive","comparison","linear-bounded-automata","undecidability","error","recurrence-relation","time-complexity","ocr","mean","ancova","boxplot","heteroscedasticity","random-forest","risk","factoring","hidden-markov-models","psychometrics","memory-hardware","word2vec","gofai","machine-learning","union-find","equality","perceptron","mindstorms","partial-order","precision-recall","sampling","ensemble","volatility-forecasting","probit","model-selection","coq","death","instrumental-variables","interpolation","numerical-algorithms","bounds","hill-climbing","missing-data","jags","distributed-systems","paired-comparisons","spss","deep-rl","social-network","q-learning","trend","linear-regression","rpart","circular-statistics","ant-colony","consensus","large-data","count-data","backtracking","randomized-algorithms","endogeneity","t-distribution","word-embedding","pathfinding","effect-size","dimensionality","hypergeometric","randomness","complexity-classes","power-analysis","contrasts","modular-arithmetic","intelligence-testing","svm","nonparametric","propensity-scores","a-star","sets","open-ai","intervals","method-of-moments","mergesort","fishers-exact","generalized-moments","permutations","cpu-pipelines","bayes","hamiltonian-path","multiplication","temporal-logic","cross-entropy","confusion-matrix","poisson-process","neuromorphic-engineering","threads","median","regression-strategies","social","alphago-zero","non-independent","weighted-graphs","pumping-lemma","kendall-tau","primitive-recursion","sentiment-analysis","bayes-theorem","concurrency","ford-fulkerson","sigmoid","arima","ordinal-data","proximal-policy-optimization","integral","data-compression","reference-question","kalman-filter","evolutionary-computing","lags","algorithm","variational-bayes","kleene-star","cart","turing-machines","self-study","aggregation","image-generation","glmm","filesystems","data-imputation","difference","euclidean-distance","power-law","extreme-value","rnn","probability-inequalities","mapping-space","deep-blue","contingency-tables","computer-games","stacks","turing-test","descriptive-statistics","ai-field","cox-model","knapsack-problem","deep-learning","biostatistics","np","church-turing-thesis","manova","ultraintelligent-machine","tukey-hsd","integers","kernel-trick","policy-gradients","autoencoders","multilevel-analysis","association-measure","goodness-of-fit","minimum-cuts","declarative-programming","feature","network-analysis","haskell","bootstrap","huffman-coding","algorithm-design","off-policy","autoregressive","definitions","nearest-neighbour","implementation","data-mining","nested-data","dropout","integer-programming","assignment-problem","vc-dimension","memory-access","bayesian","runtime-analysis","probability-theory","random-generation","unbiased-estimator","pearson-r","propositional-logic","problem-solving","autocorrelation","alpha-beta-pruning","google-cloud","binary-search-trees","model-evaluation","hierarchical-bayesian","topic-models","encryption","evolutionary-algorithms","self-play","automata","synchronization","monte-carlo-tree-search","normal-distribution","error-propagation","ddpg","stratification","prolog","gbm","learning-algorithms","constraint-programming","deviance","pytorch","rationality","planning","markov-decision-process","cyberterrorism","np-hard","enumeration","linear-temporal-logic","radix-sort","scatterplot","deepdream","iid","path-planning","substrings","modelling","prediction","interactive-proof-systems","references","simulation","boolean-algebra","order-statistics","programming-languages","interpretation","machine-translation","research","quality-control","f-test","hyper-parameters","matlab","entropy","ai-community","chi-squared","ai-development","statistical-significance","svd","exponential","multivariate-analysis","semantics","stationarity","importance-sampling","text-mining","stepwise-regression","semi-decidability","paired-data","python","data-transformation","matrix-inverse","2-sat","control-theory","database-theory","sparse","pooling","mixture","linked-lists","regular-expressions","multiple-regression","shortest-path","nasa","cars","lstm","reasoning","security","segmentation","feedback","relational-algebra","getting-started","distance","datasets","lr-k","factor-analysis","glmnet","resampling","nonlinear","text-summarization","teaching","language-design","similarity","colorings","parsing","multicollinearity","standardization","false-discovery-rate","self-awareness","zero-inflation","multi-class","philosophy","program-correctness","weighted-data","image-segmentation","wilcoxon-signed-rank","selection-problem","environment","assumptions","euclidean","human-like","time","threshold","on-policy","robust","linear-algebra","many-categories","deterministic-policy","regular-languages","optimization","semi-mdp","sas","search-algorithms","symbolic-computing","type-checking","beta-binomial","truncation","dag","exponential-smoothing","graphs","value-function","information-theory","marginal","pareto-distribution","gpu","typing","arithmetic","p-value","first-order-logic","incomplete-information","proof-assistants","robotics","databases","weak-ai","clustering","metropolis-hastings","convex-hull","conjugate-prior","robots","gam","clocks","relativization","c","atari-games","incompleteness-theorems","discount-factor","discriminant-analysis","digital-rights","aixi","algorithms","searching","generalized-least-squares","cloud-services","log-likelihood","model-based","memory-management","stochastic-policy","real-world","software","validation","inductive-datatypes","sample-size","reward-clipping","software-evaluation","binary-search","signal-processing","cdf","mgf","correlation","real-numbers","number-formats","topology","world-knowledge","control-problem","softmax","sums-of-squares","proof-techniques","education","memory-allocation","imperative-programming","sequence-analysis","multivariate-regression","convolutional-neural-networks","htm","observational-study","software-testing","covariance-matrix","bugs","action-recognition","odds","real-time","meta-regression","word-combinatorics","alphago","time-varying-covariate","image-recognition","irt","network-flow","healthcare","mips","cointegration","train","wilcoxon-mann-whitney","cluster","activation-function","term-rewriting","minimax","residuals","eligibility-traces","noise","unassisted-learning","model-free","rl-an-introduction","chinese-room-argument","turing-completeness","smoothing","information-retrieval","pushdown-automata","fuzzy-logic","bernoulli-distribution","feature-construction","feature-extraction","relation","np-complete","quantification","anomaly-detection","formal-methods","sgd","natural-language","skewness","eda","model-checking","prediction-interval","hyperparameter","libsvm","collaboration","odds-ratio","model-comparison","homotopy-type-theory","decision-theory","clinical-trials","reliability","risk-management","computer-vision","breadth-first-search","group-differences","ggplot2","programming-paradigms","fitting","operational-semantics","xor","mcmc","markov-chain","mutual-information","weka","binomial","average-case","calibration","decision-tree","linear-programming","sat-solvers","biology","polynomials","loops","experiment-design","space-complexity","proof","vif","similarities","type-inference","panel-data","compilers","logistic-regression","cohens-kappa","copula","structure","artificial-neuron","coding-theory","mutual-exclusion","data-sets","bic","measure-theory","self-driving","balanced-search-trees","approximation","sample","logic","matrix","generalization","computer-programming","unbalanced-classes","clique","logit","multidimensional-scaling","change-point","error-estimation","importance","javascript","mice","syntax","z-score","planar-graphs","reductions","decision-problem","r-squared","neat","matrices","nondeterminism","formal-grammars","bit-manipulation","t-test","ratio","conditional-expectation","dictionaries","gaussian-process","discrete-data","master-theorem","object-recognition","efficiency","gee","partial-correlation","deepdreaming","agi","pomdp","cross-validation","point-estimation","uniform","weights","gym","spiking-networks","storage","sorting","attention","cross-section","oracle-machines","watson","number-theory","constraint-satisfaction","log-linear","sem","predicting-ai-milestones","mathematical-programming","hardware","qq-plot","overfitting","training","mutation","symbolic-ai","go","computational-linguistics","fisher-information","cryptography","random-effects-model","hypercomputation","simulated-annealing","link-function","binary-data","communication-protocols","vecm","rms","likelihood","recommender-system","exponential-family","pca","hidden-markov-model","context-free","genetic-programming","repeated-measures","experience-replay","histogram","mgcv","regularization","gibbs","concepts","software-verification","granger-causality","garch","reporting","brain","eigenvalues","embedded-design","packing","game-ai","combinatorics","cpu","greedy-ai","c++","kolmogorov-complexity","ordered-logit","xgboost","elastic-net","rbm","audio-processing","multivariate-normal","constraint-satisfaction-problems","fitness-functions","covariance","knowledge-representation","random-number-generator","spanning-trees","hoare-logic","type-theory","mathematical-statistics","natural-language-processing","gaussian-mixture","asymptotics","parametric","gaming","math","econometrics","random-variable","binary","emergence","anova","consumer-product","least-squares","proofs","boltzmann-machine","return","intelligence-augmentation","finite-automata","artificial-life","aic","chomsky-hierarchy","boosting","regression","polynomial","handwritten-characters","hierarchical-clustering","architecture","graph-traversal","arrays","feature-selection","hardware-evaluation","survival","subsequences","detecting-patterns","outliers","spearman-rho","greedy-algorithms","counting","deadlocks","hash-tables","variance","maximum-entropy","bayesian-network","posterior","computation-models","parameterized-complexity","multinomial-logit","psychology","high-dimensional","central-limit-theorem","recurrent-neural-networks","embodied-cognition","pseudo-polynomial","superintelligence","variable-binding","neo-luddism","measurement","intuition","lme4-nlme","challenges","performance","combinatorial-games","statistics","edit-distance","moments","supervised-learning","search-problem","kolmogorov-smirnov","survey-sampling","protocols","proportion","k-nearest-neighbour","descriptive-complexity","diagnostic","augmented-dickey-fuller","actor-critic","deepmind","google","binary-arithmetic","prior","definition","convergence","meta-analysis","data-preprocessing","confounding","sarsa","terminology","forecasting","buchi-automata","kaplan-meier","java","human-inspired","shrinkage","matrix-decomposition","population","expected-value","beta-distribution","online-learning","probability","cyborg","nonlinear-regression","var","polynomial-time","priority-queues","ecology","k-means","computability","structured-data","ai-box","tiling","abstract-data-types","data-visualization","caret","likelihood-ratio","reinforce","metric","binary-trees","classical-ai","tensorflow","set-cover","circuits","dynamic-programming","lattices","generalized-linear-model","incremental-learning","mathematical-analysis","stata","post-hoc","papers","type-i-and-ii-errors","poisson-distribution","string-metrics","crossover","random-walk","interpreters","network-topology","neural-networks","predictive-models","artificial-intelligence","ridge-regression","bonferroni","maximum","continuous-data","ranking","accuracy","halting-problem","frequentist","lasso","intelligence","feature-engineering","curve-fitting","latent-variable","conditional-probability","data-science","facial-recognition","multi-agent-systems","language-processing","adjacency-matrix","linear-model","spectral-analysis","ai-basics","virtual-memory","curry-howard","statistical-ai","digital-circuits","ai-takeover","discrete-mathematics","multiple-comparisons","lm","induction","frequency","state-space-models","function","denotational-semantics","analysis","dqn","markov-process","sufficient-statistics","quantum-computing","quadratic-form","category-theory","encoding-scheme","satisfiability","reinforcement-learning","check-my-answer","machine-models","dataset","pdf","computer-networks","kernel-smoothing","quantile-regression","kurtosis","partial-least-squares","praxis","loop-invariants","generative-adversarial-networks","likert","mixed-model","trees","normal-forms","uncertainty","policy","value-alignment","open-source","categorical-data","os-kernel","inference","conv-neural-network","open-cog","power","complexity-theory","expert-system","formal-languages","automated-theorem-proving","theory","streaming-algorithm","online","networks","probabilistic-algorithms","bias","poker","kullback-leibler","permutation-test","z-test","co-np","chat-bots","function-approximation","value-iteration","dimensionality-reduction","agreement-statistics","backpropagation","parsers","neural-doodle","hidden-layers","interaction","games","primes","rating","books","software-architecture","asimovs-laws","computer-algebra","markov-chains","difference-in-difference","object-oriented","scikit-learn","history","lexical-recognition","spatial","process-scheduling","graph-isomorphism","computer-architecture","mythology-of-ai","knapsack-problems","heaps","3-sat","deep-network","parameterization","cellular-automata","estimation","automation","standard-deviation","chess","small-sample","categorical-encoding","cross-correlation","binning","lisp","intelligent-agent","modeling","models","convolution","ethics","loss-functions","graph-theory","trpo","lambda-calculus","error-correcting-codes","unsupervised-learning","emotional-intelligence","functional-programming","subset-sum","survey","consistency","ontology","computational-statistics","online-algorithms","stan","generative-models","fourier-transform","transfer-learning","dirichlet-distribution","clustered-standard-errors","ambiguity","expectation-maximization","board-games","scales","temporal-difference","hamming-code","excel","bioinformatics","graphical-model","pseudo-random-generators","autonomous-vehicles","cpu-cache","poisson-regression","percentage","data-structures","recursion","intraclass-correlation","standard-error","singularity","classification","mlp","ai-safety","landau-notation","search-trees","communication-complexity","thought-vectors","scipy","hypothesis-testing","overdispersion","methodology","fixed-effects-model","rare-events","weighted-sampling","mathematical-foundations","reference-request","auc","unit-root","confirmatory-factor","computational-geometry","quicksort","hebbian-learning","ai-design","treatment-effect","upper-bound","ai-a-modern-approach","sequence-modelling","keras","software-engineering"
//        };
//        tags = new HashSet<String>(Arrays.asList(TAGS));
////          tags = new HashSet();
////        tags.add("machine-learning");
////        tags.add("neural-networks");
////        tags.add("artificial-intelligence");
//    }
//
//    public static String[] tokenize(String line) {
//        return line.split(",(?=(?>[^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)", -1);
//    }
//
//    public static void process(String line) {
//
//        // QAFilter the questions based on the tags
//        String[] tokens = Merger.tokenize(line);
//
//        // Basic Filtering
//        int answerCount = Integer.parseInt(tokens[14]);
//        if (answerCount == 0) return;
//
//        String id = tokens[1];
//        String tag = tokens[13];
//        if (tags.contains(tag))
//            csQuestions.put(id, tokens);
//        else
//            randomQuestions.put(id, tokens);
//
//
//
//    }
//
//    public static void addRandomQuestionToList(String key) {
//        String[] question = randomQuestions.get(key);
//        String id = question[1];
//        csQuestions.put(id, question);
//    }
//
//    public static String merge(String line) {
//        String[] answer = Merger.tokenize(line);
//        String row = line;
//        String parentId = answer[3];
//
//        if (csQuestions.containsKey(parentId)) {
//            // Merge
//            String[] question = csQuestions.get(parentId);
//            String questionId = question[1];
//            String questionBody = question[7];
//            String questionScore = question[5];
//            String title = question[12];
//            String tags = question[13];
//
//            String answerId = answer[1];
//            String answerBody = answer[6];
//            String answerScore = answer[5];
//
//            row = questionId + "," +
//                    answerId + "," +
//                    title + "," +
//                    tags + "," +
//                    questionBody + "," +
//                    questionScore + "," +
//                    answerBody + "," +
//                    answerScore;
//
//        }
//
//        return row;
//    }
//
//    public static Stream<String> readAndMergeAnswer(String fileName) throws IOException {
//        Stream<String> stream = Files.lines(Paths.get(fileName)).skip(1);
//        return stream.parallel()
//                .filter((line) -> {
//                    String[] answer = Merger.tokenize(line);
//                    String parentId = answer[3];
//
//                    // Keep if the answer has parent id
//                    if (parentId.isEmpty())
//                        return false;
//
//                    // Keep if the corresponding question exists
//                    return csQuestions.containsKey(parentId);
//
//                })
//                .map(Merger::merge);
//    }
//
//    public static void samplingRandomQuestion() {
//
//        int csLength = csQuestions.size();
//        int randomLength = randomQuestions.size();
//
//        int target_length = DATA_COUNT - csLength;
//        target_length = Math.max(0, target_length);
//        target_length = Math.min(target_length, randomLength);
//        System.out.println("number of random questions: " + target_length);
//        List<String> keys = new ArrayList<String>(randomQuestions.keySet());
//        Collections.shuffle(keys);
//
//        keys.stream().limit(target_length).parallel().forEach(Merger::addRandomQuestionToList);
//    }
//
//
//}