package sentencesimilarity;

import au.com.bytecode.opencsv.CSVReader;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.util.*;

public class GenerateTestResults {
    public static void main(String args[])throws IOException
    {
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        ParagraphVectors vectors=WordVectorSerializer.readParagraphVectors(new File(TrainSentenceSimilarity.path+"\\vectors.txt"));
        vectors.setTokenizerFactory(t);
        vectors.getConfiguration().setIterations(1);
        System.out.println("Testing");
        CSVReader reader=new CSVReader(new FileReader(TrainSentenceSimilarity.path+"\\test.csv"));
        PrintWriter pw=new PrintWriter(new BufferedWriter(new FileWriter(TrainSentenceSimilarity.path+"\\outputs.csv")));
        int ct=0,r=0,incrrct=0,unavailable=0;
        String rd[]=null;
        do
        {
            rd=reader.readNext();
            if(rd==null || rd.length<3)
                continue;
            System.out.println(rd.length);
            System.out.println(rd[0]+" "+rd[1]+" "+rd[2]);
            String s1[]=rd;
            if(s1[0].equals("test_id"))
                continue;
            INDArray i11=null,i22=null;
            try {
                String id=s1[0];
                i11 = vectors.inferVector(s1[1]);
                i22 = vectors.inferVector(s1[2]);
                //String i11s=vecToCSV(i11),i22s=vecToCSV(i22);
                double similar=Transforms.cosineSim(i11,i22);
                pw.println(id+","+similar);

            }
            catch(Exception e)
            {
                unavailable++;
                continue;
            }
            ct++;
        }
        while(rd!=null);
        reader.close();
        pw.close();
        System.out.println("Unavailable "+unavailable);
        }
};
