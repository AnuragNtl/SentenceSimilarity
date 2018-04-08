package sentencesimilarity;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.util.*;
public class TestSentenceSimilarity {
public static void main(String args[])throws IOException
{
    TokenizerFactory t = new DefaultTokenizerFactory();
    t.setTokenPreProcessor(new CommonPreprocessor());
    ParagraphVectors vectors=WordVectorSerializer.readParagraphVectors(new File(TrainSentenceSimilarity.path+"\\vectors.txt"));
    vectors.setTokenizerFactory(t);
    vectors.getConfiguration().setIterations(1);
    System.out.println("Testing");
    BufferedReader br=new BufferedReader(new FileReader(TrainSentenceSimilarity.path+"\\clean_train_tmp.csv"));
    //PrintWriter pw=new PrintWriter(new BufferedWriter(new FileWriter(SentenceSimilarity.path+"\\pdata.csv")));
    int ct=0,r=0,incrrct=0,unavailable=0;
    String rd="";
    do
    {
        rd=br.readLine();
        if(rd==null)
            continue;
        String s1[]=rd.split(",");
        INDArray i11=null,i22=null;
        try {
            i11 = vectors.inferVector(s1[0]);
            i22 = vectors.inferVector(s1[1]);
            //String i11s=vecToCSV(i11),i22s=vecToCSV(i22);
        int i1=Integer.parseInt(s1[2]),i2=0;
        double similar=Transforms.cosineSim(i11,i22);
        /*    if(i11s.length()>0 && i22s.length()>0)
                pw.println(i11s+","+i22s+","+similar);*/
            if(similar>0.90)
            i2=1;
        if(i1==i2)
            r++;
        else
            incrrct++;
        }
        catch(Exception e)
        {
            unavailable++;
            continue;
        }
        ct++;
    }
    while(rd!=null);
    //pw.close();
    System.out.println("Unavailable "+unavailable);
    System.out.println("Total="+ct+" Correct="+r+" Incorrect="+incrrct);
    System.out.println((r*100.0)/ct);
}
private static String vecToCSV(INDArray vec)
{
    if(vec==null)
        return "";
    int len=vec.length();
    String r="";
    for(int i=0;i<len;i++)
    {
        r=r+vec.getDouble(i);
        if(i<len-1)
            r=r+",";
    }
    return r;
}
};
