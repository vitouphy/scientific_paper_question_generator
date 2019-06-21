
import org.w3c.dom.*;
import org.xml.sax.SAXException;
import xml.XmlEscape;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
//import org.

public class XmlExtractor {

//    final static int NUM_QUESTIONS = 5000;
    static int NUM_QUESTIONS = 5000;

    public static void main(String[] args) throws ParserConfigurationException, IOException, SAXException {

        // Specify the file
//        NUM_QUESTIONS = Integer.parseInt(args[0]);
        String inputFile = "/Users/vitou/Workspace/scientific_paper_question_generator/data/raw/ai.stackexchange.com/Posts.xml";
        String outputFile = "/Users/vitou/Workspace/scientific_paper_question_generator/data/intermediate/ai.stackexchange.com/ai.stackexchange.com.xml";
//        String inputFile = "/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/data/ai.stackexchange.com/Posts.xml";
//        String outputFile = "/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/analysis_001/data/ai.stackexchange.com.xml";
//        String inputFile = args[1];
//        String outputFile = args[2];
        System.out.println("Expected Number of Question: " + NUM_QUESTIONS);

        long startTime = System.nanoTime();

        // Load the file
        File fXmlFile = new File(inputFile);
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(fXmlFile);
//        doc.getDocumentElement().normalize();

        // Preparing the stream
        NodeList nodeList = doc.getElementsByTagName("row");
        Stream<Node> nodeStream = IntStream.range(0, nodeList.getLength())
                .mapToObj(nodeList::item);
        Stream<Node> nodeStream2 = IntStream.range(0, nodeList.getLength())
                .mapToObj(nodeList::item);

        // Separating Question
        QASeparator qaSeparator = new QASeparator();
        ConcurrentHashMap[] arr = qaSeparator.getQ(nodeStream);
        ConcurrentHashMap<Integer, Node> csQuestions = arr[0];
        ConcurrentHashMap<Integer, Node> randomQuestions = arr[1];
        System.out.println("Number of CS Set: " + csQuestions.size());
        System.out.println("Number of Random Set: " + randomQuestions.size());

        // Select K Questions
        selectKQuestions(NUM_QUESTIONS, csQuestions, randomQuestions);
        System.out.println("Number of Selected Question: " + csQuestions.size());
        System.out.println("\n============\n");

        List<String> outputs = qaSeparator.getA(nodeStream2, csQuestions)
                .map(row -> merge(row, csQuestions))
                .collect(Collectors.toList());

        // Output the string
        String header = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<posts>";
        String ender = "\n</posts>";

        FileWriter fw = new FileWriter(outputFile);
        fw.write(header);
        for (String out : outputs) {
            fw.write(out);
        }
        fw.write(ender);
        fw.close();

        long endTime = System.nanoTime();
        System.out.println("Execution Took: " + ((endTime - startTime) * 1e-9) + " seconds ") ;

    }

    public static void selectKQuestions(int k, ConcurrentHashMap<Integer, Node> csQuestions, ConcurrentHashMap<Integer, Node> randomQuestions) {

        int csLength = csQuestions.size();
        int randomLength = randomQuestions.size();

        // Remove element until we have K Questions
        if (csLength > k) {
            int diff = csLength - k;
            List<Integer> keys = new ArrayList<Integer>(csQuestions.keySet());
            Collections.shuffle(keys);
            for (int i=0; i<diff; i++) {
                int key = keys.get(i);
                csQuestions.remove(key);
            }
        }
        System.out.println("============\n\nSelecting K Question\n");
        System.out.println("number of chosen cs question: " + csQuestions.size());

        int target_length = k - csLength;
        target_length = Math.max(0, target_length);
        target_length = Math.min(target_length, randomLength);

        System.out.println("number of chosen random questions: " + target_length);
        if (target_length == 0) return;

        // Random Select Question from Random Set
        List<Integer> keys = new ArrayList<Integer>(randomQuestions.keySet());
        Collections.shuffle(keys);
        keys.stream().limit(target_length).parallel().forEach((key) -> {
            Element element = (Element) randomQuestions.get(key);
            int id = Integer.parseInt(element.getAttribute("Id"));
            csQuestions.put(id, element);
        });

    }

    public static String merge(Node row, ConcurrentHashMap<Integer, Node> questions) {

        String output = "";
        Element answer = (Element) row;
        int parentId = Integer.parseInt(answer.getAttribute("ParentId"));

        if (questions.containsKey(parentId)) {
            output = "\n\t<row ";
            // Merge
            Element question = (Element) questions.get(parentId);

            String questionId = question.getAttribute("Id");
            String questionBody = question.getAttribute("Body");
            String questionScore = question.getAttribute("Score");
            String title = question.getAttribute("Title");
            String tags = question.getAttribute("Tags");

            String answerId = answer.getAttribute("Id");
            String answerBody = answer.getAttribute("Body");
            String answerScore = answer.getAttribute("Score");

            output +=   "QuestionId=\"" + questionId + "\" " +
                        "AnswerId=\"" + answerId + "\" " +
                        "Title=\"" + XmlEscape.escapeXml11(title) + "\" " +
                        "Tags=\"" + XmlEscape.escapeXml11(tags) + "\" " +
                        "QuestionBody=\"" + XmlEscape.escapeXml10(clean(questionBody)) + "\" " +
                        "QuestionScore=\"" + questionScore + "\" " +
                        "AnswerBody=\"" + XmlEscape.escapeXml10(clean(answerBody)) + "\" " +
                        "AnswerScore=\"" + answerScore + "\"/>";


        }

        return output;
    }

    public static String clean(String line) {
        line = line.replaceAll("\\t", " ");
        line = line.replaceAll("\\n", " ");
        return line;
    }

    public static String convertNodeToHtml(Node node) throws TransformerException {
        Transformer t = TransformerFactory.newInstance().newTransformer();
        t.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
        StringWriter sw = new StringWriter();
        t.transform(new DOMSource(node), new StreamResult(sw));
        return sw.toString();
    }

}
