import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
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
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class TrainingSetSplitter {

    public static void main(String[] args) throws ParserConfigurationException, IOException, SAXException {

        String inputFile = "/Users/vitou/Workspace/scientific_paper_question_generator/data/intermediate/ai.stackexchange.com/ai.stackexchange.com.xml";
        String outputFolder = "/Users/vitou/Workspace/scientific_paper_question_generator/data/intermediate/ai.stackexchange.com";

        int numTrain = 5;
        int numDev = 1;
        int numTest = 2;
//        int numTrain = Integer.parseInt(args[0]);

        File fXmlFile = new File(inputFile);
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(fXmlFile);

        // Prepare the stream
        NodeList nodeList = doc.getElementsByTagName("row");
        Stream<Node> stream = IntStream.range(0, nodeList.getLength())
                .mapToObj(nodeList::item);

        // Put all row into dictionary
        ConcurrentHashMap<Integer, Node> data = new ConcurrentHashMap<Integer, Node>();
        stream.parallel().forEach((row) -> {
            Element element = (Element) row;
            int id = Integer.parseInt(element.getAttribute("AnswerId"));
            data.put(id, row);
        });

        // Shuffles
        List<Integer> keys = new ArrayList<Integer>(data.keySet());
        Collections.shuffle(keys);

        List<Integer> trainKeys = new ArrayList<Integer>();
        for (int i=0; i<numTrain; i++) {
            trainKeys.add(keys.get(i));
        }

        List<Integer> devKeys = new ArrayList<Integer>();
        for (int j=0; j<numDev; j++) {
            devKeys.add(keys.get(numTrain + j));
        }

        List<Integer> testKeys = new ArrayList<Integer>();
        for (int k=0; k<numTest; k++) {
            testKeys.add(keys.get(numTrain + numDev + k));
        }

        saveToFile(trainKeys.stream(), data, outputFolder, "train");
        saveToFile(devKeys.stream(), data, outputFolder, "dev");
        saveToFile(testKeys.stream(), data, outputFolder, "test");

    }

    public static void saveToFile(Stream<Integer> keys, ConcurrentHashMap<Integer, Node> data, String folderPath,  String type) throws IOException {

        String header = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<posts>";
        String ender = "\n</posts>";

        List<String> outputs = keys.parallel().map((key) -> {
            String output = convertNodeToString(data.get(key));
            return output;
        }).collect(Collectors.toList());

        String outputFile = Paths.get(folderPath, type + ".xml").toString();
        FileWriter fw = new FileWriter(outputFile);
        fw.write(header);
        for (String out : outputs) {
            fw.write(out);
        }
        fw.write(ender);
        fw.close();
    }

    public static String convertNodeToString(Node node) {

        Element element = (Element) node;

        String questionId = element.getAttribute("QuestionId");
        String questionBody = element.getAttribute("QuestionBody");
        String questionScore = element.getAttribute("QuestionScore");
        String title = element.getAttribute("Title");
        String tags = element.getAttribute("Tags");

        String answerId = element.getAttribute("AnswerId");
        String answerBody = element.getAttribute("AnswerBody");
        String answerScore = element.getAttribute("AnswerScore");

        String output =   "\n\t<row QuestionId=\"" + questionId + "\" " +
                "AnswerId=\"" + answerId + "\" " +
                "Title=\"" + XmlEscape.escapeXml10(title) + "\" " +
                "Tags=\"" + XmlEscape.escapeXml10(tags) + "\" " +
                "QuestionBody=\"" + XmlEscape.escapeXml10(clean(questionBody)) + "\" " +
                "QuestionScore=\"" + questionScore + "\" " +
                "AnswerBody=\"" + XmlEscape.escapeXml10(clean(answerBody)) + "\" " +
                "AnswerScore=\"" + answerScore + "\"/>";

        return output;
    }

    public static String clean(String line) {
        line = line.replaceAll("\\t", " ");
        line = line.replaceAll("\\n", " ");
        return line;
    }


}
